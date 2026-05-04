import torch
import torch.optim as optim
from torch.func import jvp
from src.model_meanflow import MeanFlowMLP
from src.dataloader import ToyDiffusionDataset


def train_meanflow(dataset_name, dim, epochs=2000, batch_size=512):

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    dataset = ToyDiffusionDataset(dataset_name, dim)
    data = torch.tensor(dataset.data, dtype=torch.float32).to(device)

    model = MeanFlowMLP(dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(epochs):

        idx = torch.randint(0, data.shape[0], (batch_size,))
        x = data[idx]

        t = torch.rand(batch_size, 1, device=device)
        r = torch.rand(batch_size, 1, device=device)

        t, r = torch.max(t, r), torch.min(t, r)

        # 50% flow matching（h=0）
        mask = torch.rand(batch_size, 1, device=device) < 0.5
        r[mask] = t[mask]

        e = torch.randn_like(x)
        z = (1 - t) * x + t * e
        v = e - x

        # JVP
        def fn(z, r, t):
            return model(z, r, t)

        u, dudt = jvp(
            fn,
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        # MeanFlow target
        u_tgt = v - (t - r) * dudt.detach()

        loss = ((u - u_tgt) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"[{dataset_name} D={dim}] step {step} loss {loss.item():.4f}")

    ckpt = f"checkpoints/meanflow_{dataset_name}_D{dim}.pt"
    torch.save(model.state_dict(), ckpt)

    return ckpt