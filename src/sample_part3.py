import torch
import matplotlib.pyplot as plt
from src.model import MLP


@torch.no_grad()
def sample_model(model, dim, steps=50, n=1000):

    device = next(model.parameters()).device
    model.eval()

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):

        t = ts[i].repeat(n)
        dt = ts[i + 1] - ts[i]

        v = model(z, t)

        z = z + v * dt

    return z