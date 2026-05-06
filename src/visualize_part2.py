import matplotlib.pyplot as plt
import os
import numpy as np

def plot_scatter(real, fake, save_path, title, lim=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(5, 5))

    plt.scatter(real[:, 0], real[:, 1], s=2, alpha=0.5, label="ground truth")
    plt.scatter(fake[:, 0], fake[:, 1], s=2, alpha=0.5, label="samples")

    plt.legend()
    plt.title(title)

    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(save_path, dpi=200)
    plt.close()