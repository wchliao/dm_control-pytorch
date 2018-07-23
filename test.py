import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.num_inputs, self.num_hidden, self.num_outputs = 5, 256, 20
        self.fc1 = nn.Linear(self.num_inputs, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.fc3 = nn.Linear(self.num_hidden, self.num_hidden)
        self.output = nn.Linear(self.num_hidden, self.num_outputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)

model = DQN()
if os.path.isfile('DQN_model'):
    model.load_state_dict(torch.load('DQN_model'))