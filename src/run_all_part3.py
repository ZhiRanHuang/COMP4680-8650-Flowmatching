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

from src.train_part3 import train_part3
from src.sample_part3 import sample_model
from src.dataloader import ToyDiffusionDataset
from src.visualize_part2 import plot_scatter
from src.model import MLP


datasets = ["swiss_roll"]
dims = [2, 8, 32]
modes = ["baseline", "rescue", "large"]

ckpt_dir = "checkpoints/part3"
os.makedirs(ckpt_dir, exist_ok=True)


def run():

    os.makedirs("part3_results", exist_ok=True)

    for dname in datasets:
        for dim in dims:
            for mode in modes:

                print(f"\n=== {dname} D={dim} MODE={mode} ===")

                # --------------------
                # checkpoint path
                # --------------------
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"{dname}_D{dim}_{mode}.pt"
                )

                # --------------------
                # train or load
                # --------------------
                if os.path.exists(ckpt_path):
                    print(f"✔ Found checkpoint: {ckpt_path}")
                    model = MLP(dim)
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                else:
                    print("✖ No checkpoint, training...")
                    ckpt_path = train_part3(dname, dim, mode)

                    model = MLP(dim)
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

                model.eval()

                # --------------------
                # sample
                # --------------------
                fake = sample_model(model, dim=dim).cpu().numpy()

                # --------------------
                # real data
                # --------------------
                dataset = ToyDiffusionDataset(dname, dim)
                real = dataset.data.numpy()

                # --------------------
                # projection
                # --------------------
                if dim != 2:
                    real = dataset.to_2d(real)
                    fake = dataset.to_2d(fake)

                # --------------------
                # save
                # --------------------
                save_path = f"part3_results/{dname}_{dim}_{mode}.png"

                plot_scatter(
                    real,
                    fake,
                    save_path,
                    f"{dname} D{dim} {mode}"
                )


if __name__ == "__main__":
    run()