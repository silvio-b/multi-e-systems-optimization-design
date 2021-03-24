import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
from torch.distributions.categorical import Categorical


class DiscreteActorNetwork(nn.Module):
    def __init__(self, alpha, input_size, max_action, fc1_size=256, fc2_size=256, n_actions=2,
                 name='actor', checkpt_dir='tmp/soft_ac'):
        super(DiscreteActorNetwork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.max_action = max_action
        self.rep_noise = 1e-6

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, self.n_actions)

        self.opt = opt.Adam(lr=alpha, params=self.parameters())
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.device = t.device('cpu')

        self.to(self.device)

    def forward(self, state):
        # state = t.tensor(state)
        prob = self.fc1(state)
        prob = f.relu(prob)
        prob = self.fc2(prob)
        prob = f.relu(prob)
        prob = self.fc3(prob)
        mu = f.softmax(prob, dim=1)

        return mu

    def sample_normal(self, state, rep=True):
        action_probabilities = self.forward(state)
        action_distribution = Categorical(action_probabilities)

        max_probability_action = t.argmax(action_probabilities, dim=-1)
        action = action_distribution.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = t.log(action_probabilities + z)

        action = action - 1
        action = action.reshape([state.shape[0], 1])

        return action, log_action_probabilities

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))