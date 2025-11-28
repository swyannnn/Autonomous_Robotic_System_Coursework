import torch
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from collections import OrderedDict
from torch.distributions.normal import Normal
from utils import ScaleLayer, DiagLinear, layer_init

class BasePPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # TODO: check how many envs are created
        # The paper expects a 1-env Gym
        # CleanRL uses vectorized envs → we must extract its single env space
        self.env = envs       # unwrap vector env
        self.build_network()

        self.space = envs.action_space
        if isinstance(self.space, spaces.Discrete):
            self.discrete_action_space = True
        elif isinstance(self.space, spaces.MultiDiscrete) and len(self.space.nvec) == 1:
            self.discrete_action_space = True
        else:
            self.discrete_action_space = False

        # Determine output dimension
        output_size = self.space.n if self.discrete_action_space else np.prod(self.space.shape)
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))

        if self.discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
        else:
            self.get_action_and_value = self._get_action_and_value_continuous

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value_discrete(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def _get_action_and_value_continuous(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def build_network(self):
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.env.action_space.n), std=0.01),
        )
    
class ParsevalPPOAgent(nn.Module):
    def __init__(self, envs, net_width=64,
                 add_diag_layer=True, activation='tanh',
                 input_scale=1, learnable_input_scale=False):
        super().__init__()

        # TODO: check how many envs are created
        # The paper expects a 1-env Gym
        # CleanRL uses vectorized envs → we must extract its single env space
        self.env = envs.envs[0]       # unwrap vector env

        self.net_width = net_width
        num_hidden = self.net_width
        self.add_diag_layer = add_diag_layer
        self.activation = activation
        self.init_gain = np.sqrt(2)
        self.net_width = net_width
        self.input_scale = input_scale
        self.learnable_input_scale = learnable_input_scale

        self.space = envs.action_space
        if isinstance(self.space, spaces.Discrete):
            self.discrete_action_space = True
        elif isinstance(self.space, spaces.MultiDiscrete) and len(self.space.nvec) == 1:
            self.discrete_action_space = True
        else:
            self.discrete_action_space = False

        # Build network exactly as in paper
        self.build_network(num_hidden, self.add_diag_layer, 
                            self.activation, self.init_gain, self.input_scale,
                            self.learnable_input_scale, self.discrete_action_space)

        # Determine output dimension
        output_size = self.space.n if self.discrete_action_space else np.prod(self.space.shape)
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))
    
        if self.discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
        else:
            self.get_action_and_value = self._get_action_and_value_continuous

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value_continuous(self, x, action=None):
        x = torch.atleast_2d(x)   # adds a batch dimension if there's only one

        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze()  # not using vector env
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def _get_action_and_value_discrete(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()    #.squeeze()  # not using vector env

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def build_network(self, num_hidden, add_diag_layer, activation, init_gain, 
                      input_scale, learnable_input_scale, discrete_action_space):
        ''' '''
        layer_name = 'linear_orthog'
        num_hidden_out = num_hidden

        actor_output_dim = self.env.action_space.n if discrete_action_space else np.prod(self.env.action_space.shape)

        if add_diag_layer:
            self.critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',nn.Tanh()),
                # Parseval regularization enforces orthogonality for all but the last layer, and the last layer must remain unrestricted to preserve expressive capacity.
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            self.actor = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), std=0.01)),
            ]))
        else:
            self.critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                (f'{activation}_2', nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            self.actor = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                (f'{activation}_2', nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), std=0.01)),
            ]))

    def parseval_reg_network(self, named_parameters):
        loss_reg = 0

        for name, param in named_parameters:
            if 'weight' in name and 'orthog' in name and param.requires_grad:
                scale = self.init_gain **2 # sqrt(2)**2
                """
                1. parseval_norm = False
                Paper proved applying regularization only to the angles (with normalized weights) 
                will make the performance was worse than standard full Parseval regularization
                so we apply regularization to the full weights 

                2. parseval_num_groups = 1
                Parseval paper mentioned for the best performance achieved by the Parseval agent, 
                the number of groups used was effectively 1,
                meaning the regularization was applied across the entire weight matrix of the layer
                """
                if param.device != torch.tensor(scale).device:
                    scale = torch.tensor(scale).to(param.device)
                loss_reg = loss_reg + torch.norm(
                    torch.matmul(param, param.t()) - scale * torch.eye(param.shape[0], device=param.device),
                    p='fro') ** 2
        return loss_reg