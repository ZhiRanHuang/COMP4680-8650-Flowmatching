import os
import torch
import numpy as np
import random

from src.visualize_part6 import plot_scatter_part6

# ----------------------------
# FIX SEED
# ----------------------------
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

from src.train_meanflow import train_meanflow
from src.sample_meanflow import sample_meanflow
from src.dataloader import ToyDiffusionDataset
from src.visualize_part2 import plot_scatter


datasets = ["swiss_roll", "gaussians", "circles"]
steps_list = [1, 2, 5]
dim = 32


# ============================
# GLOBAL SCALE
# ============================
def compute_global_lim():
    all_data = []

    for dname in datasets:
        real = np.load(f"{dname}_D{dim}.npy")[:, :2]
        all_data.append(real)

    all_data = np.vstack(all_data)
    return np.max(np.abs(all_data))


def run():

    os.makedirs("part6_results", exist_ok=True)

    global_lim = compute_global_lim()

    for dname in datasets:

        print(f"\n=== Training MeanFlow: {dname} D={dim} ===")
        ckpt = train_meanflow(dname, dim)

        model = torch.load(ckpt, map_location="cpu")

        from src.model_meanflow import MeanFlowMLP
        model_obj = MeanFlowMLP(dim)
        model_obj.load_state_dict(model)
        model_obj.eval()

        real = np.load(f"{dname}_D{dim}.npy")[:, :2]

        for steps in steps_list:

            fake = sample_meanflow(model_obj, dim, steps=steps).detach().numpy()
            fake = fake[:, :2]

            save_path = f"part6_results/{dname}_meanflow_{steps}step.png"

            plot_scatter_part6(
                real,
                fake,
                save_path,
                f"{dname} MeanFlow {steps} step",
                global_lim
            )


if __name__ == "__main__":
    run()