import torch
import os
from torch.utils.data import DataLoader
from src.model import MLP
from src.dataloader import ToyDiffusionDataset


def train_part3(dataset_name, dim, mode="baseline", steps=25000):

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = ToyDiffusionDataset(dataset_name, dim)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    # -----------------------
    # model scaling
    # -----------------------
    hidden = 256
    if mode == "large":
        hidden = 1024
    elif mode == "rescue":
        hidden = 512

    model = MLP(dim).to(device)

    # overwrite hidden size if needed
    model.hidden = hidden

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints/part3", exist_ok=True)

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

        # -----------------------
        # optional rescue tricks
        # -----------------------
        if mode == "rescue":
            v_pred = v_pred / (dim ** 0.5)
            v_target = v_target / (dim ** 0.5)

        loss = ((v_pred - v_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0:
            print(f"[{dataset_name} D={dim} {mode}] step {step} loss {loss.item():.4f}")

    os.makedirs("checkpoints/part3", exist_ok=True)

    ckpt = f"checkpoints/part3/{dataset_name}_D{dim}_{mode}.pt"
    torch.save(model.state_dict(), ckpt)

    return ckpt