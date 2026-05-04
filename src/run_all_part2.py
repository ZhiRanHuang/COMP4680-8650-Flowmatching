import os
import torch

from src.train_part2 import train_model
from src.sample_part2 import sample_model
from src.dataloader import ToyDiffusionDataset
from src.visualize_part2 import plot_scatter
from src.model import MLP


datasets = ["swiss_roll", "gaussians", "circles"]
dims = [2, 8, 32]

pred_types = ["x", "v"]
loss_types = ["x", "v"]


def run_all():

    os.makedirs("part2_results", exist_ok=True)

    for dname in datasets:
        for dim in dims:
            for pred in pred_types:
                for loss in loss_types:

                    print(f"\n=== Running {dname} D={dim} {pred}-{loss} ===")

                    # -------------------------
                    # 1. train model
                    # -------------------------
                    ckpt = train_model(dname, dim, pred, loss)

                    # -------------------------
                    # 2. load dataset
                    # -------------------------
                    dataset = ToyDiffusionDataset(dname, dim)

                    real = dataset.data.numpy()

                    real_2d = dataset.to_2d(real)

                    # -------------------------
                    # 3. load model
                    # -------------------------
                    model = MLP(dim)
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

                    # -------------------------
                    # 4. sample
                    # -------------------------
                    fake = sample_model(model, dim=dim).cpu().numpy()

                    fake_2d = dataset.to_2d(fake)

                    # -------------------------
                    # 5. plot
                    # -------------------------
                    save_path = f"part2_results/{dname}_D{dim}_{pred}_{loss}.png"

                    plot_scatter(
                        real_2d,
                        fake_2d,
                        save_path,
                        f"{dname} D{dim} {pred}-{loss}"
                    )


if __name__ == "__main__":
    run_all()