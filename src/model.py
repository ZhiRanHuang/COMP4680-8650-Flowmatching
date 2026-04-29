import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.time_dim = 128
        self.hidden = 256

        self.net = nn.Sequential(
            nn.Linear(dim + self.time_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, dim),
        )

    def time_embed(self, t):
        device = t.device
        half = self.time_dim // 2

        freqs = torch.exp(
            -torch.arange(half, device=device) * math.log(10000) / (half - 1)
        )

        args = t[:, None] * freqs[None, :]

        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)