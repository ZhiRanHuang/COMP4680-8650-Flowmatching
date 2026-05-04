import torch

import torch.nn as nn

class MeanFlowMLP(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, r, t):
        h = t - r
        x = torch.cat([z, t, h], dim=1)
        return self.net(x)