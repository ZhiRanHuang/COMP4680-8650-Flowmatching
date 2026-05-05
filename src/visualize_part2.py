import matplotlib.pyplot as plt
import os


def plot_scatter(real, fake, save_path, title):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(5, 5))

    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.5, label="ground truth")
    plt.scatter(fake[:, 0], fake[:, 1], s=2, alpha=0.5, label="samples")

    plt.legend()
    plt.axis("equal")
    plt.title(title)

    plt.savefig(save_path, dpi=200)
    plt.close()