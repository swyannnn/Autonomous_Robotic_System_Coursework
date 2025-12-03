import gymnasium as gym
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import numpy as np

def make_env(env_id: str, capture_video: bool, run_name: str, max_episode_steps: int = None) -> callable:
    """
    Utility function for creating a Gym environment with optional video capture and episode length limit.
    Args:
        env_id (str): The environment ID for Gym.
        capture_video (bool): Whether to capture video of the environment.
        run_name (str): The name of the run for video saving.
        max_episode_steps (int, optional): Maximum number of steps per episode. Defaults to None.
    Returns:
        function: A function that creates and returns the Gym environment when called.
    """
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class DiagLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = x * c + b`.
        Multiplies by a scalar and adds a bias.

        Based on Linear layer code from Pytorch

    Args:
        in_features: size of each input sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.

        - Output: :math:`(*, H_{in})` . Same as input.

    Attributes:
        weight: the learnable weights of the module of shape H_{in}
                All initialized to 1

        bias:   the learnable bias of the module of shape H_{in}
                All initialized to 0

    """
    __constants__ = ['in_features']
    def __init__(self, in_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, input: Tensor) -> Tensor:
        return input * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class ScaleLayer(nn.Module):
    """
    Layer that scales the input by a learnable or fixed factor.
    """
    def __init__(self, init_value=1.0, learnable=False):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        self.scale.requires_grad = learnable

    def forward(self, input):
        return input * self.scale

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module :
    """
    Initialize a layer with orthogonal weights and constant bias.
    Args:
        layer: The layer to initialize.
        std: Standard deviation for the orthogonal initialization.
        bias_const: Constant value for bias initialization.
    Returns:
        The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer