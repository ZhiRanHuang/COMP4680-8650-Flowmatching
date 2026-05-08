import torch

@torch.no_grad()
def sample_model(model, dim, n=1000, steps=50):

    device = next(model.parameters()).device
    model.eval()

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):

        t = ts[i].expand(n)
        t_next = ts[i + 1]

        dt = (t_next - ts[i])

        v = model(z, t)

        # only safety, NOT scaling geometry
        v = torch.nan_to_num(v, 0.0)

        scale = v.norm(dim=1).mean()
        dt_eff = dt / (1.0 + scale)

        z = z + v * dt_eff

    return z