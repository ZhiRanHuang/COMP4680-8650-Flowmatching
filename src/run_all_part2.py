import os
import torch
import numpy as np
import random

# ----------------------------
# FIX SEED
# ----------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

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
                    # checkpoint path
                    # -------------------------
                    ckpt_path = (
                        f"checkpoints/part2/"
                        f"{dname}_D{dim}_{pred}_{loss}.pt"
                    )

                    # -------------------------
                    # 1. train model if needed
                    # -------------------------
                    if os.path.exists(ckpt_path):

                        print(f"Checkpoint exists: {ckpt_path}")
                        print("Skipping training...")

                    else:

                        print("Training model...")
                        ckpt = train_model(dname, dim, pred, loss)

                        # rename/move checkpoint if needed
                        if ckpt != ckpt_path:
                            os.rename(ckpt, ckpt_path)

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
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

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