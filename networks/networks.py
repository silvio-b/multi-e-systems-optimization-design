import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[256, 256], init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)
        self.ln1 = nn.LayerNorm(hidden_size[0])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = t.cat([state, action], 1)
        x = self.ln1(f.relu(self.linear1(x)))
        x = self.ln2(f.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space, action_scaling_coef, hidden_dim=[256, 256],
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = t.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = t.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = t.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = t.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= t.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = t.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)


class SoftQNetworkDiscrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=None, init_w=3e-3):
        super(SoftQNetworkDiscrete, self).__init__()

        if hidden_size is None:
            hidden_size = [256, 256]

        self.input_layer = nn.Linear(num_inputs, hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(len(hidden_size)):
            self.hidden_layers.append(nn.Linear(hidden_size[k], hidden_size[k]))

        self.output_layer = nn.Linear(hidden_size[len(hidden_size)-1], num_actions)

        # self.output_layer.weight.data.fill_(-10)
        # self.output_layer.bias.data.fill_(-10)
        # self.ln1 = nn.LayerNorm(hidden_size[0])
        # self.ln2 = nn.LayerNorm(hidden_size[1])

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        q_value = f.relu(self.input_layer(state))

        for layer in self.hidden_layers:
            q_value = f.relu(layer(q_value))

        q_value = self.output_layer(q_value)

        return q_value


class PolicyNetworkDiscrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=None,
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetworkDiscrete, self).__init__()

        if hidden_size is None:
            hidden_size = [256, 256]

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.input_layer = nn.Linear(num_inputs, hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(len(hidden_size)):
            self.hidden_layers.append(nn.Linear(hidden_size[k], hidden_size[k]))

        self.output_layer = nn.Linear(hidden_size[len(hidden_size)-1], num_actions)
        # self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)
        # self.output_layer.weight.data.fill_(-100)
        # self.output_layer.bias.data.fill_(-100)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)


        # self.action_scale = t.FloatTensor(
        #     action_scaling_coef * (action_space.high - action_space.low) / 2.)
        # self.action_bias = t.FloatTensor(
        #     action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        prob = f.relu(self.input_layer(state))

        for layer in self.hidden_layers:
            prob = f.relu(layer(prob))

        prob = t.softmax(self.output_layer(prob), dim=1)
        return prob

    def sample(self, state):
        action_probabilities = self.forward(state)
        max_probability_action = t.argmax(action_probabilities, dim=-1)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = t.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action
