import torch
from torch.utils.data import DataLoader
from src.dataloader import ToyDiffusionDataset
from src.model import MLP

device = "cuda"

# ---------------------------
# data
# ---------------------------
dataset = ToyDiffusionDataset("swiss_roll", dim=2)
loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)

data_iter = iter(loader)

# ---------------------------
# model
# ---------------------------
model = MLP(dim=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# training loop
# ---------------------------
for step in range(20000):

    try:
        x = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x = next(data_iter)

    x = x.to(device)

    # time
    t = torch.rand(x.shape[0], device=device)

    # noise
    eps = torch.randn_like(x)

    # interpolation
    z = (1 - t[:, None]) * x + t[:, None] * eps

    # target velocity
    v_target = eps - x

    # prediction
    v_pred = model(z, t)

    # loss
    loss = ((v_pred - v_target) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

print("training done")