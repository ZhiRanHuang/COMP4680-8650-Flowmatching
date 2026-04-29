import torch
import matplotlib.pyplot as plt
from src.model import MLP


# -----------------------
# sampling function
# -----------------------
@torch.no_grad()
def sample(model, n=1000, steps=50):
    device = next(model.parameters()).device
    model.eval()

    z = torch.randn(n, 2, device=device)

    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t = ts[i].repeat(n)
        t = t.clamp(1e-5, 1.0)

        dt = ts[i + 1] - ts[i]

        v = model(z, t)
        z = z + v * dt

    return z


# -----------------------
# main script
# -----------------------
if __name__ == "__main__":
    device = "cuda"
    model = MLP(dim=2).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    samples = sample(model).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=2)
    plt.axis("equal")
    plt.title("Generated samples")
    plt.show()