import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

datasets = ["swiss_roll", "gaussians", "circles"]

# MeanFlow steps
mf_steps = [1, 2, 5]

# Flow Matching steps
fm_steps = [50, 20, 10]


def merge():

    os.makedirs("part6_compare", exist_ok=True)

    for d in datasets:

        fig, axes = plt.subplots(3, 2, figsize=(8, 12))

        for i in range(3):

            # -------------------------
            # MeanFlow
            # -------------------------
            mf_path = f"part6_results/{d}_meanflow_{mf_steps[i]}step.png"
            mf_img = mpimg.imread(mf_path)

            axes[i, 0].imshow(mf_img)
            axes[i, 0].set_title(f"MeanFlow {mf_steps[i]} step")
            axes[i, 0].axis("off")

            # -------------------------
            # Flow Matching
            # -------------------------
            fm_path = f"part6_fm_results/{d}_fm_{fm_steps[i]}step.png"
            fm_img = mpimg.imread(fm_path)

            axes[i, 1].imshow(fm_img)
            axes[i, 1].set_title(f"Flow Matching {fm_steps[i]} steps")
            axes[i, 1].axis("off")

        plt.suptitle(f"{d} Comparison", fontsize=16)

        save_path = f"part6_compare/{d}_compare.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")


if __name__ == "__main__":
    merge()