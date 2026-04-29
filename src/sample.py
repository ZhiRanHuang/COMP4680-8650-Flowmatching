import torch
import matplotlib.pyplot as plt
from src.model import MLP
import os

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
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_path = os.path.join(

        "checkpoints",

        "model_swiss_roll_D2.pt"

    )

    # -----------------------
    # model
    # -----------------------
    model = MLP(dim=2).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # -----------------------
    # sampling
    # -----------------------
    samples = sample(model).cpu().numpy()

    # -----------------------
    # plot
    # -----------------------
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=2)
    plt.axis("equal")
    plt.title("Generated samples")
    plt.show()