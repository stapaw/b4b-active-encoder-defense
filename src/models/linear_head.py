import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MapperNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x