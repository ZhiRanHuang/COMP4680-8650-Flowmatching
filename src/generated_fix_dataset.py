import numpy as np

from src.dataloader import ToyDiffusionDataset

datasets = ["swiss_roll", "gaussians", "circles"]

dim = 32

for d in datasets:
    ds = ToyDiffusionDataset(d, dim)
    np.save(f"{d}_D{dim}.npy", ds.data.numpy())