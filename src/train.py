import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataloader import ToyDiffusionDataset
from src.model import MLP

# ----------------------------
# config
# ----------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

datasets = ["swiss_roll", "gaussians", "circles"]
dim = 2

batch_size = 1024
lr = 1e-3
steps = 25000

# ----------------------------
# training loop
# ----------------------------
def train_one(dataset_name):

    dataset = ToyDiffusionDataset(dataset_name, dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    model = MLP(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints/part1", exist_ok=True)

    for step in range(steps):

        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(device)

        t = torch.rand(x.shape[0], device=device).clamp(1e-5, 1 - 1e-5)
        eps = torch.randn_like(x)

        z = (1 - t[:, None]) * x + t[:, None] * eps

        v_target = eps - x
        v_pred = model(z, t)

        loss = ((v_pred - v_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 1000 == 0:
            print(f"[{dataset_name}] step {step} | loss {loss.item():.6f}")

    ckpt_path = f"checkpoints/part1/{dataset_name}_D2.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"Saved: {ckpt_path}")


# ----------------------------
# run all datasets
# ----------------------------
if __name__ == "__main__":
    for d in datasets:
        train_one(d)