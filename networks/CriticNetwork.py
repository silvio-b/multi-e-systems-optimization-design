import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt


class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, input_size, n_actions, hidden_size=None,
                 name='critic', checkpt_dir='tmp/soft_ac'):
        super(CriticNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.input_layer = nn.Linear(self.input_size+self.n_actions, self.hidden_size[0])

        self.hidden_layers = nn.ModuleList()
        for k in range(len(self.hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_size[k], self.hidden_size[k + 1]))

        self.q = nn.Linear(self.hidden_size[k + 1], 1)

        self.opt = opt.Adam(lr=learning_rate, params=self.parameters())
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = f.relu(self.input_layer(t.cat([state, action], dim=1)))

        for layer in self.hidden_layers:
            action_value = f.relu(layer(action_value))

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))