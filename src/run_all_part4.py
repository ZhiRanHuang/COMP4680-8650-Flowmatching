import os
import torch

from src.dataloader import ToyDiffusionDataset
from src.model import MLP
from src.sample_part4 import sample_model
from src.visualize_part2 import plot_scatter  # 或你自己的plot


# ----------------------------
# BEST MODEL PATH
# ----------------------------
CKPT_PATH = "checkpoints/part2/swiss_roll_D2_v_v.pt"


# ----------------------------
# steps list
# ----------------------------
STEPS_LIST = [1, 2, 5, 10, 20, 50, 100, 200]


def load_model():

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    ckpt = torch.load(CKPT_PATH, map_location=device)

    # 自动推断 dim
    dim = ckpt["net.10.bias"].shape[0]

    model = MLP(dim).to(device)
    model.load_state_dict(ckpt)

    return model, dim, device


def run():

    os.makedirs("part4_results", exist_ok=True)

    model, dim, device = load_model()

    dataset = ToyDiffusionDataset("swiss_roll", dim=2)
    real = dataset.data.numpy()

    for steps in STEPS_LIST:

        print(f"Running steps={steps}")

        fake = sample_model(model, n=2000, steps=steps).cpu().numpy()

        save_path = f"part4_results/swiss_roll_steps_{steps}.png"

        plot_scatter(
            real,
            fake,
            save_path,
            f"Swiss Roll | Euler steps={steps}"
        )


if __name__ == "__main__":
    run()