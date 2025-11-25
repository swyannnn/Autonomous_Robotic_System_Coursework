import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from collections import OrderedDict
from torch.distributions.normal import Normal
from utils import ScaleLayer, DiagLinear


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BaseAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

        self.discrete_action_space = isinstance(self.env.env.action_space, gym.spaces.Discrete)
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
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def create_block_diag_matrix(block_size, num_blocks):
    # Create a single block of 1s
    block = torch.ones(block_size, block_size)

    # Create a list of blocks
    blocks = [block] * num_blocks

    # Use torch.block_diag to create a block diagonal matrix
    block_diag_matrix = torch.block_diag(*blocks)

    return block_diag_matrix

class ParsevalAgent(nn.Module):
    def __init__(self, envs, lambda_parseval=0.0001, net_width=64,
                 add_diag_layer=True, activation='tanh', weight_init='orthogonal'):
        super().__init__()

        # The paper expects a 1-env Gym
        # CleanRL uses vectorized envs → we must extract its single env space
        env = envs.envs[0]       # unwrap vector env

        # Build AgentNetworks exactly as in paper
        self.networks = AgentNetworks(
            env=env,
            network_type='mlp',
            weight_init=weight_init,
            init_gain=None,
            layer_norm=False,
            layer_norm_no_params=False,
            net_width=net_width,
            activation=activation,
            parseval_reg=lambda_parseval,
            add_diag_layer=add_diag_layer,
            input_scale=1,
            learnable_input_scale=False,
            discrete_action_space=True,
        )

        self.lambda_parseval = lambda_parseval
        self.net_width = net_width

    def get_value(self, x):
        return self.networks.get_value(x)

    def get_action_and_value(self, x, action=None):
        return self.networks.get_action_and_value(x, action)

    def parseval_reg_network(self, named_parameters):
        loss_reg = 0

        for name, param in named_parameters:
            if 'weight' in name and 'orthog' in name and param.requires_grad:
                # print(self.init_gain)
                if self.init_gain is None:
                    scale = 2 # sqrt(2)**2
                else:
                    scale = self.init_gain **2

                if self.parseval_norm:
                    temp_par = param / torch.norm(param, dim=1).view(-1,1)
                else:
                    temp_par = param

                # weight matrices right multiply by their inputs
                if self.parseval_num_groups == 1:
                    if temp_par.device != torch.tensor(scale).device:
                        scale = torch.tensor(scale).to(temp_par.device)
                    loss_reg = loss_reg + torch.norm(
                        torch.matmul(temp_par, temp_par.t()) - scale * torch.eye(temp_par.shape[0], device=temp_par.device),
                        p='fro') ** 2
                elif self.parseval_num_groups > 1:
                    if self.net_width % self.parseval_num_groups != 0:
                        raise AssertionError(
                            f'net_width ({self.net_width}) has to be divisible by parseval_num_groups ({self.parseval_num_groups})')

                    neuron_group_size = self.net_width // self.parseval_num_groups
                    mask_matrix = create_block_diag_matrix(neuron_group_size, self.parseval_num_groups)

                    loss_reg = loss_reg + torch.norm(
                        mask_matrix * torch.matmul(temp_par, temp_par.t()) - scale * torch.eye(
                            temp_par.shape[0]),
                        p='fro') ** 2
        return loss_reg

class AgentNetworks(nn.Module):
    def __init__(self, env, init_gain=None,
                 layer_norm=False, layer_norm_no_params=False,
                 net_width=64, activation=None, add_diag_layer=False,
                 input_scale=1, learnable_input_scale=False,
                 discrete_action_space=None):
        super().__init__()
        self.init_gain = init_gain
        self.net_width = net_width
        self.activation = activation
        self.add_diag_layer = add_diag_layer
        self.input_scale = input_scale
        self.learnable_input_scale = learnable_input_scale

        self.discrete_action_space = discrete_action_space

        # for k, v in locals().items():
        #     print(k, v)

        num_hidden = net_width

        self.actor_mean, self.critic = self.build_network(env, num_hidden, layer_norm, layer_norm_no_params,
                                                            add_diag_layer, activation, init_gain, input_scale,
                                                            learnable_input_scale, discrete_action_space)
        output_size = env.action_space.n if discrete_action_space else np.prod(env.action_space.shape)
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))

        if discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
        else:
            self.get_action_and_value = self._get_action_and_value_continuous

    def get_value(self, x):
        return self.critic(x)


    def _get_action_and_value_continuous(self, x, action=None):
        # hmm add tsallis entropy?
        x = torch.atleast_2d(x)   # adds a batch dimension if there's only one

        action_mean = self.actor_mean(x)
        # print('x', x.shape)
        # print('action_mean', action_mean.shape)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze()  # not using vector env
            # print(action)
        # print(probs.log_prob(action))
        # print(probs.log_prob(action).shape)
        # quit()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


    def _get_action_and_value_discrete(self, x, action=None):
        # x = torch.atleast_2d(x)   # adds a batch dimension if there's only one

        logits = self.actor_mean(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()    #.squeeze()  # not using vector env

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


    def build_network(self, env, num_hidden, add_diag_layer, activation, init_gain, 
                      input_scale, learnable_input_scale, discrete_action_space):
        ''' '''
        layer_name = 'linear_orthog'
        num_hidden_out = num_hidden

        actor_output_dim = env.action_space.n if discrete_action_space else np.prod(env.action_space.shape)

        if add_diag_layer:
            critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            actor_mean = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), std=0.01)),
            ]))
        else:
            critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                (f'{activation}_2', nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            actor_mean = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), num_hidden_out), std=init_gain)),
                (f'{activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=init_gain)),
                (f'{activation}_2', nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), std=0.01)),
            ]))
        return actor_mean, critic
