# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import adaptive piecewise layers
from non_uniform_piecewise_layers.adaptive_piecewise_mlp import AdaptivePiecewiseMLP
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear
from non_uniform_piecewise_layers.adaptive_piecewise_conv import AdaptivePiecewiseConv2d


def weight_init(layer: torch.nn.modules):
    if isinstance(layer, (AdaptivePiecewiseLinear, AdaptivePiecewiseConv2d)):
        # Adaptive layers have their own initialization
        pass
    elif isinstance(layer, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(layer.weight.data, gain)
        if hasattr(layer.bias, 'data'): layer.bias.data.fill_(0.0)


def ln_activ(x: torch.Tensor, activ: Callable):
    x = F.layer_norm(x, (x.shape[-1],))
    return activ(x)


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str='elu', num_points: int=3, position_range=(-1, 1)):
        super().__init__()
        
        # Replace standard Linear layers with AdaptivePiecewiseLinear
        self.l1 = AdaptivePiecewiseLinear(input_dim, hdim, num_points=num_points, position_range=position_range)
        self.l2 = AdaptivePiecewiseLinear(hdim, hdim, num_points=num_points, position_range=position_range)
        self.l3 = AdaptivePiecewiseLinear(hdim, output_dim, num_points=num_points, position_range=position_range)

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, x: torch.Tensor):
        y = ln_activ(self.l1(x), self.activ)
        y = ln_activ(self.l2(y), self.activ)
        return self.l3(y)
        
        
    def move_smoothest(self, weighted: bool=True):
        """Call move_smoothest on all adaptive layers in this MLP.
        
        Args:
            weighted (bool): Whether to use weighted error calculation. Default is True.
            
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            success = True
            # Move smoothest points in all layers
            success = self.l1.move_smoothest(weighted=weighted) and success
            success = self.l2.move_smoothest(weighted=weighted) and success
            success = self.l3.move_smoothest(weighted=weighted) and success
        return success


class Encoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, pixel_obs: bool,
        num_bins: int=65, zs_dim: int=512, za_dim: int=256, zsa_dim: int=512, hdim: int=512, activ: str='elu',
        num_points: int=3, position_range=(-1, 1)):
        super().__init__()
        
        self.num_points = num_points
        self.position_range = position_range
        self.pixel_obs = pixel_obs
        
        if pixel_obs:
            self.zs = self.cnn_zs
            # Replace standard Conv2d layers with AdaptivePiecewiseConv2d
            self.zs_cnn1 = AdaptivePiecewiseConv2d(state_dim, 32, 3, stride=2, num_points=num_points, position_range=position_range)
            self.zs_cnn2 = AdaptivePiecewiseConv2d(32, 32, 3, stride=2, num_points=num_points, position_range=position_range)
            self.zs_cnn3 = AdaptivePiecewiseConv2d(32, 32, 3, stride=2, num_points=num_points, position_range=position_range)
            self.zs_cnn4 = AdaptivePiecewiseConv2d(32, 32, 3, stride=1, num_points=num_points, position_range=position_range)
            self.zs_lin = AdaptivePiecewiseLinear(1568, zs_dim, num_points=num_points, position_range=position_range)
        else:
            self.zs = self.mlp_zs
            self.zs_mlp = BaseMLP(state_dim, zs_dim, hdim, activ, num_points=num_points, position_range=position_range)

        # Replace standard Linear layer with AdaptivePiecewiseLinear
        self.za = AdaptivePiecewiseLinear(action_dim, za_dim, num_points=num_points, position_range=position_range)
        self.zsa = BaseMLP(zs_dim + za_dim, zsa_dim, hdim, activ, num_points=num_points, position_range=position_range)
        self.model = AdaptivePiecewiseLinear(zsa_dim, num_bins + zs_dim + 1, num_points=num_points, position_range=position_range)

        self.zs_dim = zs_dim

        self.activ = getattr(F, activ)
        self.apply(weight_init)


    def forward(self, zs: torch.Tensor, action: torch.Tensor):
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], 1))


    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        dzr = self.model(zsa)
        return dzr[:,0:1], dzr[:,1:self.zs_dim+1], dzr[:,self.zs_dim+1:] # done, zs, reward


    def cnn_zs(self, state: torch.Tensor):
        state = state/255. - 0.5
        zs = self.activ(self.zs_cnn1(state))
        zs = self.activ(self.zs_cnn2(zs))
        zs = self.activ(self.zs_cnn3(zs))
        zs = self.activ(self.zs_cnn4(zs)).reshape(state.shape[0], -1)
        return ln_activ(self.zs_lin(zs), self.activ)


    def mlp_zs(self, state: torch.Tensor):
        return ln_activ(self.zs_mlp(state), self.activ)
        
    def move_smoothest(self, weighted: bool=True):
        """Call move_smoothest on all adaptive layers in this encoder.
        
        Args:
            weighted (bool): Whether to use weighted error calculation. Default is True.
            
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            success = True
            
            # Move smoothest points in all layers
            if self.pixel_obs:
                # CNN path
                success = self.zs_cnn1.move_smoothest(weighted=weighted) and success
                success = self.zs_cnn2.move_smoothest(weighted=weighted) and success
                success = self.zs_cnn3.move_smoothest(weighted=weighted) and success
                success = self.zs_cnn4.move_smoothest(weighted=weighted) and success
                success = self.zs_lin.move_smoothest(weighted=weighted) and success
            else:
                # MLP path
                success = self.zs_mlp.move_smoothest(weighted=weighted) and success
            
            success = self.za.move_smoothest(weighted=weighted) and success
            success = self.zsa.move_smoothest(weighted=weighted) and success
            success = self.model.move_smoothest(weighted=weighted) and success
            
        return success


class Policy(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float=10, zs_dim: int=512, hdim: int=512, activ: str='relu',
                 num_points: int=3, position_range=(-1, 1)):
        super().__init__()
        
        # Replace standard MLP with AdaptivePiecewiseMLP
        # Create width list for AdaptivePiecewiseMLP [input_size, hidden_size, output_size]
        width = [zs_dim, hdim, action_dim]
        self.policy = AdaptivePiecewiseMLP(width, num_points=num_points, position_range=position_range)
        
        self.activ = partial(F.gumbel_softmax, tau=gumbel_tau) if discrete else torch.tanh
        self.discrete = discrete


    def forward(self, zs: torch.Tensor):
        pre_activ = self.policy(zs)
        action = self.activ(pre_activ)
        return action, pre_activ


    def act(self, zs: torch.Tensor):
        action, _ = self.forward(zs)
        return action
        
        
    def move_smoothest(self, weighted: bool=True):
        """Call move_smoothest on all adaptive layers in this policy.
        
        Args:
            weighted (bool): Whether to use weighted error calculation. Default is True.
            
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            # Move smoothest points in the policy MLP
            success = self.policy.move_smoothest(weighted=weighted)
        return success


class Value(nn.Module):
    def __init__(self, zsa_dim: int=512, hdim: int=512, activ: str='elu', num_points: int=3, position_range=(-1, 1)):
        super().__init__()

        class ValueNetwork(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hdim: int=512, activ: str='elu', 
                         num_points: int=3, position_range=(-1, 1)):
                super().__init__()
                
                # Replace standard MLP with AdaptivePiecewiseMLP
                # Create width list for AdaptivePiecewiseMLP [input_size, hidden_size]
                self.q1 = AdaptivePiecewiseMLP([input_dim, hdim, hdim], num_points=num_points, position_range=position_range)
                self.q2 = AdaptivePiecewiseLinear(hdim, output_dim, num_points=num_points, position_range=position_range)

                self.activ = getattr(F, activ)
                self.apply(weight_init)

            def forward(self, zsa: torch.Tensor):
                zsa = ln_activ(self.q1(zsa), self.activ)
                return self.q2(zsa)
                
            def move_smoothest(self, weighted: bool=True):
                """Call move_smoothest on all adaptive layers in this ValueNetwork.
                
                Args:
                    weighted (bool): Whether to use weighted error calculation. Default is True.
                    
                Returns:
                    bool: True if points were successfully moved in all layers, False otherwise.
                """
                with torch.no_grad():
                    success = True
                    # Move smoothest points in all layers
                    success = self.q1.move_smoothest(weighted=weighted) and success
                    success = self.q2.move_smoothest(weighted=weighted) and success
                return success

        self.q1 = ValueNetwork(zsa_dim, 1, hdim, activ, num_points=num_points, position_range=position_range)
        self.q2 = ValueNetwork(zsa_dim, 1, hdim, activ, num_points=num_points, position_range=position_range)


    def forward(self, zsa: torch.Tensor):
        return torch.cat([self.q1(zsa), self.q2(zsa)], 1)
        
        
    def move_smoothest(self, weighted: bool=True):
        """Call move_smoothest on all adaptive layers in this Value network.
        
        Args:
            weighted (bool): Whether to use weighted error calculation. Default is True.
            
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            success = True
            # Move smoothest points in both value networks
            success = self.q1.move_smoothest(weighted=weighted) and success
            success = self.q2.move_smoothest(weighted=weighted) and success
        return success