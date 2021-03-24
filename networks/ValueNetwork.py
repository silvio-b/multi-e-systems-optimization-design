import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt

class ValueNetwork(nn.Module):
    def __init__(self, learning_rate, input_size, hidden_size, name='value', checkpt_dir='tmp/soft_ac'):
        super(ValueNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.input_layer = nn.Linear(self.input_size, self.hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(len(self.hidden_size) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_size[k], self.hidden_size[k + 1]))
        self.v = nn.Linear(self.hidden_size[k + 1], 1)

        self.opt = opt.Adam(lr=learning_rate, params=self.parameters())
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state = t.tensor(state)
        state_value = f.relu(self.input_layer(state))

        for layer in self.hidden_layers:
            state_value = f.relu(layer(state_value))

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))