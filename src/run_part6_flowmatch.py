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

from src.model import MLP
from src.sample_part2 import sample_model
from src.visualize_part2 import plot_scatter

datasets = ["swiss_roll", "gaussians", "circles"]
steps_list = [10, 20, 50]
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

    os.makedirs("part6_fm_results", exist_ok=True)

    device = "cpu"

    global_lim = compute_global_lim()

    for dname in datasets:

        print(f"\n=== Flow Matching: {dname} D=32 ===")

        # -------------------------
        # load dataset
        # -------------------------
        real = np.load(f"{dname}_D{dim}.npy")[:, :2]

        # -------------------------
        # load checkpoint
        # -------------------------
        ckpt_path = f"checkpoints/part2/{dname}_D32_v_v.pt"

        model = MLP(dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # -------------------------
        # sampling
        # -------------------------
        for steps in steps_list:

            print(f"Sampling {steps} steps...")

            fake = sample_model(
                model,
                dim=dim,
                steps=steps
            ).cpu().numpy()

            fake_2d = fake[:, :2]

            save_path = f"part6_fm_results/{dname}_fm_{steps}step.png"

            plot_scatter(
                real,
                fake_2d,
                save_path,
                f"{dname} FM {steps} steps",
                lim=global_lim
            )


if __name__ == "__main__":
    run()