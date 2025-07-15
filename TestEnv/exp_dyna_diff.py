import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo2.npy'  # Set the specific demo to compare
raw_base_dir = os.path.join("..", "DataEngine", "data", "super_dqn")
perturbed_base_dir = os.path.join("data", "super_dqn_sae")

# Ego vehicle is the first row of 2D observation reshaped as [N, num_vehicles, num_features]
num_features = 5  # e.g., existance, x, y, vx, vy
ego_index = 0

feature_names = ["existance", "x", "y", "vx", "vy"]
feature_indices = list(range(num_features))
num_rows = len(feature_indices) + 1  # +1 for label comparison

fig, axes = plt.subplots(num_rows, len(scenarios), figsize=(18, 12), sharex=True)
axes = axes.reshape(num_rows, len(scenarios))

for si, scenario in enumerate(scenarios):
    raw_obs_path = os.path.join(raw_base_dir, scenario, "obs", demo_file)
    perturbed_obs_path = os.path.join(perturbed_base_dir, scenario, "obs", demo_file)
    raw_label_path = os.path.join(raw_base_dir, scenario, "labels", demo_file)
    perturbed_label_path = os.path.join(perturbed_base_dir, scenario, "labels", demo_file)

    # Check for file availability
    if not all(os.path.exists(p) for p in [raw_obs_path, perturbed_obs_path, raw_label_path, perturbed_label_path]):
        for row in range(num_rows):
            axes[row, si].axis("off")
        continue

    # Load obs data
    raw_obs = np.load(raw_obs_path).reshape(-1, 5, num_features)
    perturbed_obs = np.load(perturbed_obs_path).reshape(-1, 5, num_features)

    # Load label data
    raw_labels = np.load(raw_label_path)
    perturbed_labels = np.load(perturbed_label_path)

    # Plot ego features
    for fi, fname in enumerate(feature_names):
        ax = axes[fi, si]
        raw_vals = raw_obs[:, ego_index, fi]
        pert_vals = perturbed_obs[:, ego_index, fi]

        ax.plot(raw_vals, linestyle='--', alpha=0.7, label='Raw')
        ax.plot(pert_vals, linestyle='-', alpha=0.7, label='Perturbed')
        ax.set_title(f"{scenario} | {fname}")
        ax.grid(True)
        if fi == 0 and si == 0:
            ax.legend(fontsize="x-small")

    # Plot label comparison with fixed y-axis
    ax = axes[-1, si]
    ax.plot(raw_labels, linestyle='--', alpha=0.7, label="Raw Action")
    ax.plot(perturbed_labels, linestyle='-', alpha=0.7, label="Perturbed Action")
    ax.set_title(f"{scenario} | Actions")
    ax.set_ylim(-0.5, 4.5)  # <- Fixed y-axis range for actions
    ax.grid(True)
    if si == 0:
        ax.legend(fontsize="x-small")

fig.suptitle(f"Ego Feature + Action Label Comparison (Raw vs Perturbed)\nDemo: {demo_file}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
