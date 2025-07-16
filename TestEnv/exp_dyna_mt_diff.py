import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Maximum Survival Frames for each scenario
max_survival_frames = {
    "highway": 40,
    "merge": 15,
    "roundabout": 11
}

# Configuration
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo1.npy'
raw_base_dir = os.path.join("..", "DataEngine", "data", "super_dqn")
mt_base_dir = os.path.join("data_dyna_test")
test_folders = ['data_mr23', 'data_mr24', 'data_mr34']

num_features = 5
ego_index = 0
feature_names = ["existance", "x", "y", "vx", "vy"]
num_rows = len(feature_names) + 1  # +1 for action labels

# Store MT results
mt_results = []

# Create subplots
fig, axes = plt.subplots(num_rows, len(scenarios), figsize=(18, 12), sharex=True)
axes = axes.reshape(num_rows, len(scenarios))

for si, scenario in enumerate(scenarios):
    # Load original data
    raw_obs_path = os.path.join(raw_base_dir, scenario, "obs", demo_file)
    raw_label_path = os.path.join(raw_base_dir, scenario, "labels", demo_file)

    if not all(os.path.exists(p) for p in [raw_obs_path, raw_label_path]):
        for row in range(num_rows):
            axes[row, si].axis("off")
        continue

    raw_obs = np.load(raw_obs_path).reshape(-1, 5, num_features)
    raw_labels = np.load(raw_label_path)
    survival_threshold = max_survival_frames[scenario]
    survived_raw = len(raw_labels) == survival_threshold

    # Plot original data
    for fi, fname in enumerate(feature_names):
        axes[fi, si].plot(raw_obs[:, ego_index, fi], linestyle='--', alpha=0.7, label='Raw')
    axes[-1, si].plot(raw_labels, linestyle='--', alpha=0.7, label="Raw Action")

    # Iterate over test folders
    for test_folder in test_folders:
        pert_obs_path = os.path.join(mt_base_dir, test_folder, "super_dqn_sae", scenario, "obs", demo_file)
        pert_label_path = os.path.join(mt_base_dir, test_folder, "super_dqn_sae", scenario, "labels", demo_file)

        if not all(os.path.exists(p) for p in [pert_obs_path, pert_label_path]):
            continue

        pert_obs = np.load(pert_obs_path).reshape(-1, 5, num_features)
        pert_labels = np.load(pert_label_path)
        survived_pert = len(pert_labels) == survival_threshold

        # Plot perturbed data
        for fi, fname in enumerate(feature_names):
            axes[fi, si].plot(pert_obs[:, ego_index, fi], alpha=0.6, label=test_folder)
        axes[-1, si].plot(pert_labels, alpha=0.6, label=test_folder)

        # Determine MT Result
        if survived_raw == survived_pert:
            mt_result = "MT Success"
        else:
            mt_result = "MT Fail"

        # Store result
        mt_results.append({
            "Scenario": scenario,
            "TestFolder": test_folder,
            "OrigLength": len(raw_labels),
            "TransLength": len(pert_labels),
            "OrigSurvival": "Survived" if survived_raw else "Failed",
            "TransSurvival": "Survived" if survived_pert else "Failed",
            "MT_Result": mt_result
        })

    # Formatting
    for fi in range(num_rows):
        axes[fi, si].set_title(f"{scenario} | {'Actions' if fi == num_rows-1 else feature_names[fi]}")
        axes[fi, si].grid(True)
        if fi == num_rows - 1:
            axes[fi, si].set_ylim(-0.5, 4.5)
        if si == 0 and fi == 0:
            axes[fi, si].legend(fontsize="x-small")

# Final layout
fig.suptitle(f"Ego Feature + Action Label Comparison Across MR Tests\nDemo: {demo_file}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Print and optionally save MT results
df_mt = pd.DataFrame(mt_results)
print("\n=== Metamorphic Testing Results ===")
print(df_mt.to_string(index=False))

# Save results to CSV
df_mt.to_csv("mt_results.csv", index=False)
print("\nResults saved to mt_results.csv")
