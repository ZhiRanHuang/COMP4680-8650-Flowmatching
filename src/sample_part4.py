import torch


@torch.no_grad()
def sample_model(model, n=2000, steps=50):

    device = next(model.parameters()).device
    model.eval()

    dim = model.net[-1].out_features  # 自动识别输出维度

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t = ts[i].repeat(n)

        dt = ts[i + 1] - ts[i]

        v = model(z, t)
        z = z + v * dt

    return z