import matplotlib.pyplot as plt
from src.dataloader import ToyDiffusionDataset

datasets = ["swiss_roll", "gaussians", "circles"]

for name in datasets:
    # D=2
    ds2 = ToyDiffusionDataset(name=name, dim=2)
    x2 = ds2.data.numpy()

    plt.figure()
    plt.scatter(x2[:, 0], x2[:, 1], s=2)
    plt.title(f"{name} (D=2)")
    plt.savefig(f"{name}_d2.png")
    plt.close()

    # D=32 -> project back
    ds32 = ToyDiffusionDataset(name=name, dim=32)
    x32 = ds32.data.numpy()
    x32_2d = ds32.to_2d(x32)

    plt.figure()
    plt.scatter(x32_2d[:, 0], x32_2d[:, 1], s=2)
    plt.title(f"{name} (D=32 projected)")
    plt.savefig(f"{name}_d32.png")
    plt.close()