import torch
from torch.utils.data import DataLoader
from src.dataloader import ToyDiffusionDataset
from src.model import MLP
import csv
import os

# ---------------------------
# device setup
# ---------------------------
device = "mps"

# ---------------------------
# config (IMPORTANT for report)
# ---------------------------
dataset_name = "swiss_roll"
dim = 2

batch_size = 1024
lr = 1e-3
steps = 25000

# ---------------------------
# data
# ---------------------------
dataset = ToyDiffusionDataset(dataset_name, dim=dim)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
data_iter = iter(loader)

# ---------------------------
# model
# ---------------------------
model = MLP(dim=dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------------
# logging setup
# ---------------------------
os.makedirs("logs", exist_ok=True)
log_file = f"logs/log_{dataset_name}_D{dim}.csv"

with open(log_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])

# ---------------------------
# training loop
# ---------------------------
for step in range(steps):

    # restart dataloader if needed
    try:
        x = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x = next(data_iter)

    x = x.to(device)

    # sample time
    t = torch.rand(x.shape[0], device=device).clamp(1e-5, 1 - 1e-5)

    # noise
    eps = torch.randn_like(x)

    # interpolation (forward process)
    z = (1 - t[:, None]) * x + t[:, None] * eps

    # v-pred target
    v_target = eps - x

    # model prediction
    v_pred = model(z, t)

    # loss (v-loss)
    loss = ((v_pred - v_target) ** 2).mean()

    # optimization
    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---------------------------
    # logging
    # ---------------------------
    if step % 50 == 0:
        print(f"[{dataset_name} D={dim}] step {step} | loss {loss.item():.6f}")

        with open(log_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([step, loss.item()])

# ---------------------------
# save model
# ---------------------------
os.makedirs("checkpoints", exist_ok=True)

ckpt_path = f"checkpoints/model_{dataset_name}_D{dim}.pt"
torch.save(model.state_dict(), ckpt_path)

print("training done")
print(f"model saved to {ckpt_path}")
print(f"log saved to {log_file}")