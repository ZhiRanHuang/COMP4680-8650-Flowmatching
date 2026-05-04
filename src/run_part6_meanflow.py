import os
import torch
from src.train_meanflow import train_meanflow
from src.sample_meanflow import sample_meanflow
from src.dataloader import ToyDiffusionDataset
from src.visualize_part2 import plot_scatter


datasets = ["swiss_roll", "gaussians", "circles"]
steps_list = [1, 2, 5]
dim = 32


def run():

    os.makedirs("part6_results", exist_ok=True)
    for dname in datasets:
        print(f"\n=== Training MeanFlow: {dname} D={dim} ===")
        ckpt = train_meanflow(dname, dim)

        model = torch.load(ckpt, map_location="cpu")
        model_obj = None

        from src.model_meanflow import MeanFlowMLP
        model_obj = MeanFlowMLP(dim)
        model_obj.load_state_dict(model)
        model_obj.eval()

        dataset = ToyDiffusionDataset(dname, dim)
        real = dataset.data[:, :2]

        for steps in steps_list:

            fake = sample_meanflow(model_obj, dim, steps=steps).detach().numpy()
            fake = fake[:, :2]

            save_path = f"part6_results/{dname}_meanflow_{steps}step.png"

            plot_scatter(
                real,
                fake,
                save_path,
                f"{dname} MeanFlow {steps} step"
            )


if __name__ == "__main__":
    run()