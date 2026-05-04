import torch

def sample_meanflow(model, dim, steps=1, n_samples=2000):
    device = next(model.parameters()).device
    z = torch.randn(n_samples, dim, device=device)
    if steps == 1:
        r = torch.zeros(n_samples, 1, device=device)
        t = torch.ones(n_samples, 1, device=device)
        u = model(z, r, t)
        x = z - u
        return x

    # few-step
    ts = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n_samples, 1)
        r = ts[i + 1].expand(n_samples, 1)
        u = model(z, r, t)
        z = z - (t - r) * u

    return z