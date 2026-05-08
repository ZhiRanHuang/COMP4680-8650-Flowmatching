import os
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter_part6(real, fake, save_path, title, global_lim):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    real = np.nan_to_num(real)
    fake = np.nan_to_num(fake)

    plt.figure(figsize=(6, 6))

    plt.scatter(real[:, 0], real[:, 1],
                s=3, alpha=0.6,
                label="ground truth", zorder=3)

    plt.scatter(fake[:, 0], fake[:, 1],
                s=3, alpha=0.6,
                label="samples", zorder=2)

    plt.legend()
    plt.title(title)

    # ✔ FIXED GLOBAL SCALE
    plt.xlim(-global_lim, global_lim)
    plt.ylim(-global_lim, global_lim)

    # ✔ IMPORTANT: prevent ellipse distortion
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()