import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo1.npy'  # Set the specific demo to compare
raw_base_dir = os.path.join("..", "DataEngine", "data", "super_dqn")
perturbed_base_dir = os.path.join("data", "super_dqn_sae")

# Ego vehicle is the first row of 2D observation reshaped as [N, num_vehicles, num_features]
num_features = 5  # e.g., existance, x, y, vx, vy
ego_index = 0

feature_names = ["existance", "x", "y", "vx", "vy"]
feature_indices = list(range(num_features))

fig, axes = plt.subplots(len(feature_indices), len(scenarios), figsize=(18, 10), sharex=True)
axes = axes.reshape(len(feature_indices), len(scenarios))

for fi, fname in enumerate(feature_names):
    for si, scenario in enumerate(scenarios):
        raw_obs_path = os.path.join(raw_base_dir, scenario, "obs", demo_file)
        perturbed_obs_path = os.path.join(perturbed_base_dir, scenario, "obs", demo_file)

        ax = axes[fi, si]

        if not os.path.exists(raw_obs_path) or not os.path.exists(perturbed_obs_path):
            ax.axis("off")
            continue

        raw_obs = np.load(raw_obs_path).reshape(-1, 5, num_features)
        perturbed_obs = np.load(perturbed_obs_path).reshape(-1, 5, num_features)

        raw_vals = raw_obs[:, ego_index, fi]
        pert_vals = perturbed_obs[:, ego_index, fi]

        ax.plot(raw_vals, linestyle='--', alpha=0.7, label='Raw')
        ax.plot(pert_vals, linestyle='-', alpha=0.7, label='Perturbed')
        ax.set_title(f"{scenario} | {fname}")
        ax.grid(True)

        if fi == 0 and si == 0:
            ax.legend(fontsize="x-small")

fig.suptitle(f"Ego Feature Comparison Over Time (Raw vs Perturbed)\nDemo: {demo_file}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
