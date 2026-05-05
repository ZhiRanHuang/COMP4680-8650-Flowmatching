import torch


@torch.no_grad()
def sample_model(model, dim, n=1000, steps=50):

    device = next(model.parameters()).device
    model.eval()

    z = torch.randn(n, dim, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):

        t = ts[i].repeat(n).to(device)
        t_next = ts[i + 1]

        dt = t_next - ts[i]

        raw = model(z, t)

        # convert prediction → velocity field
        if raw.shape == z.shape:
            # assume v-pred directly
            v = raw
        else:
            raise ValueError("Model output shape mismatch")

        z = z + v * dt

    return z