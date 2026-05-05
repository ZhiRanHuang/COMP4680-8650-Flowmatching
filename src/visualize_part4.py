import matplotlib.pyplot as plt

def plot_step(real, fake, save_path, title):

    plt.figure(figsize=(5, 5))

    plt.scatter(real[:, 0], real[:, 1], s=2, label="ground truth")

    plt.scatter(fake[:, 0], fake[:, 1], s=2, label="samples")

    plt.legend()

    plt.axis("equal")

    plt.title(title)

    plt.savefig(save_path)

    plt.close()