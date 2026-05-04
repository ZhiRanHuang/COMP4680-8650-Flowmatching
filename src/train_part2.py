import torch
import os
from torch.utils.data import DataLoader
from src.model import MLP
from src.dataloader import ToyDiffusionDataset


def train_model(dataset_name, dim, pred_type, loss_type, steps=25000):

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = ToyDiffusionDataset(dataset_name, dim)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    model = MLP(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints/part2", exist_ok=True)

    model.train()

    for step in range(steps):

        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(device)

        # t in (0,1)
        t = torch.rand(x.shape[0], device=device).clamp(1e-5, 1 - 1e-5)

        eps = torch.randn_like(x)

        # forward process
        z = (1 - t[:, None]) * x + t[:, None] * eps

        # model prediction (raw output depends on pred_type)
        raw = model(z, t)

        # -------------------------
        # parameterization conversion
        # -------------------------
        if pred_type == "x":
            x_pred = raw
            v_pred = eps - x_pred   # convert x -> v

        elif pred_type == "v":
            v_pred = raw
            x_pred = z + (1 - t[:, None]) * v_pred  # convert v -> x

        # -------------------------
        # loss space
        # -------------------------
        if loss_type == "x":
            loss = ((x_pred - x) ** 2).mean()

        elif loss_type == "v":
            v_target = eps - x
            loss = ((v_pred - v_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0:
            print(f"[{dataset_name} D={dim} {pred_type}-{loss_type}] step {step} loss {loss.item():.4f}")

    ckpt_path = f"checkpoints/part2/{dataset_name}_D{dim}_{pred_type}_{loss_type}.pt"
    torch.save(model.state_dict(), ckpt_path)

    return ckpt_path