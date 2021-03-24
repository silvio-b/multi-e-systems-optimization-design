import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    def __init__(self, learning_rate, input_size, max_action, hidden_size=None, n_actions=2,
                 name='actor', checkpt_dir='tmp/soft_ac'):
        super(ActorNetwork, self).__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.max_action = max_action
        self.rep_noise = 1e-6

        self.input_layer = nn.Linear(self.input_size, self.hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(len(self.hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_size[k], self.hidden_size[k + 1]))
        self.mu = nn.Linear(self.hidden_size[k + 1], self.n_actions)
        self.sigma = nn.Linear(self.hidden_size[k + 1], self.n_actions)

        self.opt = opt.Adam(lr=learning_rate, params=self.parameters())
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state = t.tensor(state)
        prob = f.relu(self.input_layer(state))

        for layer in self.hidden_layers:
            prob = f.relu(layer(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = t.clamp(sigma, min=self.rep_noise, max=1)  # ??

        return mu, sigma

    def sample_normal(self, state, rep=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if rep:
            actions = probabilities.rsample()  # Add some noise
            # actions = (1 - (-1)) * t.rand(state.shape[0], 1).to(self.device) + (-1)
        else:
            actions = probabilities.sample()

        action = t.tanh(actions)*t.tensor(self.max_action).to(self.device)
        log_prob = probabilities.log_prob(actions)
        log_prob -= t.log(1-action.pow(2)+self.rep_noise)  # ???????
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))