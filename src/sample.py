import os
import torch
import matplotlib.pyplot as plt

from src.model import MLP
from src.dataloader import ToyDiffusionDataset

# ----------------------------
# config
# ----------------------------
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

datasets = ["swiss_roll", "gaussians", "circles"]
dim = 2
steps = 50

ckpt_dir = "checkpoints/part1"
out_dir = "part1_3.2_results"
os.makedirs(out_dir, exist_ok=True)


# ----------------------------
# Euler sampling
# ----------------------------
@torch.no_grad()
def sample(model, n=2000):

    z = torch.randn(n, dim, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t = torch.full((n,), ts[i].item(), device=device)
        dt = ts[i + 1] - ts[i]

        v = model(z, t)
        z = z + v * dt

    return z


# ----------------------------
# plot
# ----------------------------
def plot(real, fake, title, save_path):

    plt.figure(figsize=(5, 5))

    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.5, label="ground truth")
    plt.scatter(fake[:, 0], fake[:, 1], s=2, alpha=0.5, label="samples")

    plt.legend()
    plt.axis("equal")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ----------------------------
# run
# ----------------------------
def run():

    for d in datasets:

        print(f"Evaluating {d}")

        dataset = ToyDiffusionDataset(d, dim)
        real = dataset.data.numpy()

        ckpt_path = f"{ckpt_dir}/{d}_D2.pt"

        model = MLP(dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        fake = sample(model).cpu().numpy()

        save_path = f"{out_dir}/{d}_D2.png"

        plot(real, fake, f"{d} Flow Matching (50 steps)", save_path)

        print(f"Saved {save_path}")


if __name__ == "__main__":
    run()