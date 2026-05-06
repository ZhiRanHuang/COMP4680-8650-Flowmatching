import numpy as np

def get_global_lim():
    datasets = ["swiss_roll", "gaussians", "circles"]
    dim = 32
    all_data = []
    for d in datasets:
        real = np.load(f"{d}_D{dim}.npy")[:, :2]
        all_data.append(real)
    all_data = np.vstack(all_data)
    return np.max(np.abs(all_data))