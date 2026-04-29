import torch
from torch.utils.data import DataLoader
from src.dataloader import ToyDiffusionDataset
from src.model import MLP
import csv
import os

# ---------------------------
# device
# ---------------------------
device = "cuda"

print("Using device:", device)

# ---------------------------
# data
# ---------------------------
dataset_name = "swiss_roll"
dim = 2

dataset = ToyDiffusionDataset(dataset_name, dim=dim)
loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)
data_iter = iter(loader)

# ---------------------------
# model
# ---------------------------
model = MLP(dim=dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# logging setup
# ---------------------------
log_file = f"logs_{dataset_name}_D{dim}.csv"
os.makedirs("logs", exist_ok=True)
log_file = f"logs/logs_{dataset_name}_D{dim}.csv"

with open(log_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])

# ---------------------------
# training
# ---------------------------
for step in range(25000):

    try:
        x = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x = next(data_iter)

    x = x.to(device)

    # time
    t = torch.rand(x.shape[0], device=device).clamp(1e-5, 1-1e-5)

    # noise
    eps = torch.randn_like(x)

    # interpolation
    z = (1 - t[:, None]) * x + t[:, None] * eps

    # target velocity (v-pred)
    v_target = eps - x

    # prediction
    v_pred = model(z, t)

    # loss
    loss = ((v_pred - v_target) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---------------------------
    # logging
    # ---------------------------
    if step % 50 == 0:
        print(f"[{dataset_name} D={dim}] step {step} | loss {loss.item():.4f}")

        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([step, loss.item()])

print("training done")