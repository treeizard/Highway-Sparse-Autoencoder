import os
import numpy as np

# Load data
def load_npy_pairs(obs_dir, label_dir):
    obs_files = sorted([f for f in os.listdir(obs_dir) if f.endswith('.npy')])
    X, y = [], []
    for fname in obs_files:
        obs = np.load(os.path.join(obs_dir, fname))
        label = np.load(os.path.join(label_dir, fname))
        X.append(obs)
        y.append(label)
    return np.vstack(X), np.concatenate(y)