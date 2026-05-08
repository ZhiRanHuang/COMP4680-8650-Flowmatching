import matplotlib.pyplot as plt
import os
import numpy as np

def plot_scatter(real, fake, save_path, title):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    real = np.nan_to_num(real)
    fake = np.nan_to_num(fake)

    plt.figure(figsize=(6, 6))

    plt.scatter(real[:, 0], real[:, 1], s=3, alpha=0.5, label="ground truth", zorder=3)
    plt.scatter(fake[:, 0], fake[:, 1], s=3, alpha=0.5, label="samples", zorder=2)

    plt.legend()
    plt.title(title)

    min_xy = real.min(axis=0)
    max_xy = real.max(axis=0)

    margin = 0.1 * (max_xy - min_xy + 1e-6)

    plt.xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
    plt.ylim(min_xy[1] - margin[1], max_xy[1] + margin[1])

    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()