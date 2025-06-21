import torch
import torch.nn as nn

class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        q_values = self.net(x)
        return torch.argmax(q_values, dim=1)