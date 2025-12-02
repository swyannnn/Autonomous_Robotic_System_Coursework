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
        self.env = envs       # unwrap vector env

        self.space = envs.action_space
        if isinstance(self.space, spaces.Discrete):
            self.discrete_action_space = True
        elif isinstance(self.space, spaces.MultiDiscrete) and len(self.space.nvec) == 1:
            self.discrete_action_space = True
        else:
            self.discrete_action_space = False

        if self.discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
            self.actor_output_dim = self.space.n
        else:
            self.get_action_and_value = self._get_action_and_value_continuous
            self.rpo_alpha = 0.5  # RPO algorithm, specifically for continuous action space 
            self.actor_output_dim = np.prod(self.space.shape)

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.actor_output_dim))
        self.build_network()

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value_discrete(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            action = action.squeeze()  # not using vector env
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def _get_action_and_value_continuous(self, x, action=None):
        x = torch.atleast_2d(x)
        
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze(-1)
        else:
            # sample again to add stochasticity, for the policy update
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(action_mean.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
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
            layer_init(nn.Linear(64, self.actor_output_dim), std=0.01),
        )

class PPOAgent(nn.Module):
    def __init__(self, envs, net_width=64,
                 add_diag_layer=True, activation='tanh',
                 input_scale=1, learnable_input_scale=False):
        super().__init__()

        self.env = envs
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

        if self.discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
            self.actor_output_dim = self.space.n
        else:
            self.get_action_and_value = self._get_action_and_value_continuous
            self.actor_output_dim = np.prod(self.space.shape)
            self.rpo_alpha = 0.5  # RPO algorithm, specifically for continuous action space 

        self.actor_logstd = nn.Parameter(torch.zeros(1, self.actor_output_dim))
        self.build_network(num_hidden)

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value_continuous(self, x, action=None):
        x = torch.atleast_2d(x)
        
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze(-1)
        else:
            # sample again to add stochasticity, for the policy update
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(action_mean.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def _get_action_and_value_discrete(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()    #.squeeze()  # not using vector env
        action = action.squeeze(-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def build_network(self, num_hidden):
        ''' '''
        num_hidden_out = num_hidden

        actor_output_dim = self.env.action_space.n if self.discrete_action_space else np.prod(self.env.action_space.shape)

        if self.add_diag_layer:
            layer_name = 'linear_orthog'
            self.critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(self.input_scale, self.learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=self.init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{self.activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=self.init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{self.activation}_2',nn.Tanh()),
                # Parseval regularization enforces orthogonality for all but the last layer, and the last layer must remain unrestricted to preserve expressive capacity.
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            self.actor = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(self.input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=self.init_gain)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{self.activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=self.init_gain)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{self.activation}_2',nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), std=0.01)),
            ]))
        else:
            layer_name = 'linear'
            self.critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(self.input_scale, self.learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=self.init_gain)),
                (f'{self.activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=self.init_gain)),
                (f'{self.activation}_2', nn.Tanh()),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), std=1.0)),
            ]))
            self.actor = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(self.input_scale, self.learnable_input_scale)),
                (f'{layer_name}_1', layer_init(nn.Linear(np.array(self.env.observation_space.shape).prod(), num_hidden_out), std=self.init_gain)),
                (f'{self.activation}_1', nn.Tanh()),
                (f'{layer_name}_2', layer_init(nn.Linear(num_hidden, num_hidden_out), std=self.init_gain)),
                (f'{self.activation}_2', nn.Tanh()),
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
    
    def get_log_quantities(self):
        logged_values = {}
        with torch.no_grad():
            ### Save states for testing later

            ### gradient norm
            actor_grads = []
            for name, param in self.actor.named_parameters():
                if param.requires_grad and param.grad is not None:
                    actor_grads.append(param.grad.view(-1))

            if len(actor_grads) > 0:
                actor_grads = torch.cat(actor_grads)
                actor_norm = torch.linalg.vector_norm(actor_grads, ord=2).item()
            else:
                actor_norm = torch.tensor(0.0)

            logged_values['actor_grad_norm'] = actor_norm

            # --- CRITIC GRADIENT NORM ---
            critic_grads = []
            for name, param in self.critic.named_parameters():
                if param.requires_grad and param.grad is not None:
                    critic_grads.append(param.grad.view(-1))

            if len(critic_grads) > 0:
                critic_grads = torch.cat(critic_grads)
                critic_norm = torch.linalg.vector_norm(critic_grads, ord=2).item()
            else:
                critic_norm = torch.tensor(0.0)

            logged_values['critic_grad_norm'] = critic_norm

            ### params and gradients singular values
            actor_gradient_singular_values = []
            actor_singular_values = []
            for name, param in self.actor.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # ---- SKIP IF GRADIENT IS NONE ----
                    if param.grad is None:
                        actor_gradient_singular_values.append(None)
                    else:
                        _, grad_sv, _ = torch.svd(torch.atleast_2d(param.grad))
                        actor_gradient_singular_values.append(grad_sv.cpu().numpy())

                    # ---- PARAM ALWAYS EXISTS ----
                    _, param_sv, _ = torch.svd(torch.atleast_2d(param))
                    actor_singular_values.append(param_sv.cpu().numpy())

            critic_gradient_singular_values = []
            critic_singular_values = []
            for name, param in self.critic.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # ---- SKIP IF GRADIENT IS NONE ----
                    if param.grad is None:
                        critic_gradient_singular_values.append(None)
                    else:
                        _, grad_sv, _ = torch.svd(torch.atleast_2d(param.grad))
                        critic_gradient_singular_values.append(grad_sv.cpu().numpy())

                    # ---- PARAM ALWAYS EXISTS ----
                    _, param_sv, _ = torch.svd(torch.atleast_2d(param))
                    critic_singular_values.append(param_sv.cpu().numpy())

            logged_values['actor_grad_singular_values'] = actor_gradient_singular_values
            logged_values['actor_param_singular_values'] = actor_singular_values
            logged_values['critic_grad_singular_values'] = critic_gradient_singular_values
            logged_values['critic_param_singular_values'] = critic_singular_values

            ### parameter and bias norms
            actor_weight_norms = []
            actor_bias_norms = []
            for name, param in self.actor.named_parameters():
                # print("ACTOR", name)
                if ('weight' in name or 'bias' in name) and param.requires_grad:
                    # print("ACTOR2", name)
                    if 'weight' in name:
                        param = torch.atleast_2d(param)
                        weight_norm = torch.linalg.matrix_norm(param, ord='fro')
                        actor_weight_norms.append(weight_norm.item())
                    elif 'bias' in name:
                        bias_norm = torch.linalg.vector_norm(param, ord=2)
                        actor_bias_norms.append(bias_norm.item())

            critic_weight_norms = []
            critic_bias_norms = []
            for name, param in self.critic.named_parameters():
                # print("ACTOR", name)
                if ('weight' in name or 'bias' in name) and param.requires_grad:
                    # print("ACTOR2", name)
                    if 'weight' in name:
                        param = torch.atleast_2d(param)
                        weight_norm = torch.linalg.matrix_norm(param, ord='fro')
                        critic_weight_norms.append(weight_norm.item())
                    elif 'bias' in name:
                        bias_norm = torch.linalg.vector_norm(param, ord=2)
                        critic_bias_norms.append(bias_norm.item())

            logged_values['actor_weight_norms'] = actor_weight_norms
            logged_values['actor_bias_norms'] = actor_bias_norms
            logged_values['critic_weight_norms'] = critic_weight_norms
            logged_values['critic_bias_norms'] = critic_bias_norms

            ### stable rank
            # compute stable rank, ratio of squared frobenius norm to squared spectral norm
            actor_stable_ranks = []
            for name, param in self.actor.named_parameters():
                # print("ACTOR", name)

                if 'weight' in name and 'linear' in name:
                    param = torch.atleast_2d(param)
                    stable_rank = (torch.linalg.matrix_norm(param, ord='fro') / torch.linalg.matrix_norm(param, ord=2)) **2
                    actor_stable_ranks.append(stable_rank.item())

            critic_stable_ranks = []
            for name, param in self.critic.named_parameters():
                if 'weight' in name and 'linear' in name:
                    param = torch.atleast_2d(param)
                    stable_rank = (torch.linalg.matrix_norm(param, ord='fro') / torch.linalg.matrix_norm(param, ord=2)) **2
                    critic_stable_ranks.append(stable_rank.item())

            logged_values['actor_matrix_stable_rank'] = actor_stable_ranks
            logged_values['critic_matrix_stable_rank'] = critic_stable_ranks

            ### cosine similarity
            actor_cosine_sim_per_layer = []
            for name, param in self.actor.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # only consider the angle between vectors
                    # we normalize the weights row-wise and then regularize towards identity
                    normed_param = torch.nn.functional.normalize(param, dim=1)
                    cosine_sim = torch.norm(
                        torch.matmul(normed_param, normed_param.t()) - torch.eye(param.shape[0], device=normed_param.device),
                        p='fro') ** 2  # removed the diagonal entries

                    cosine_sim = cosine_sim / (param.shape[0]**2 - param.shape[0])  # avg over entries
                    actor_cosine_sim_per_layer.append(cosine_sim.item())

            critic_cosine_sim_per_layer = []
            for name, param in self.critic.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # only consider the angle between vectors
                    # we normalize the weights row-wise and then regularize towards identity
                    normed_param = torch.nn.functional.normalize(param, dim=1)
                    cosine_sim = torch.norm(
                        torch.matmul(normed_param, normed_param.t()) - torch.eye(param.shape[0], device=normed_param.device),
                        p='fro') ** 2  # removed the diagonal entries
                    if (param.shape[0] ** 2 - param.shape[0]) != 0:
                        cosine_sim = cosine_sim / (param.shape[0]**2 - param.shape[0])  # avg over entries
                    critic_cosine_sim_per_layer.append(cosine_sim.item())
            logged_values['actor_cosine_sim'] = actor_cosine_sim_per_layer
            logged_values['critic_cosine_sim'] = critic_cosine_sim_per_layer

        return logged_values