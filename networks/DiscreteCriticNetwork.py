import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt


class DiscreteCriticNetwork(nn.Module):
    def __init__(self, beta, input_size, n_actions, fc1_size=256, fc2_size=256,
                 name='critic', checkpt_dir='tmp/soft_ac'):
        super(DiscreteCriticNetwork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.q = nn.Linear(self.fc2_size, self.n_actions)

        self.opt = opt.Adam(lr=beta, params=self.parameters())
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.device = t.device('cpu')

        self.to(self.device)

    def forward(self, state):
        action_value = self.fc1(state)
        action_value = f.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = f.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))