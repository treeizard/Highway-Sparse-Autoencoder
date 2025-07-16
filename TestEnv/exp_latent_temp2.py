import os
import numpy as np
import pandas as pd

# -------------------------------------
# Config
# -------------------------------------
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo1.npy'
latent_dim = 1024
model_name = "super_dqn_sae"
mr_name = "MR23"
MAX_T = 40

original_dir = os.path.join("data_original", model_name)
mr_dir = os.path.join("data_dyna_test", f"data_{mr_name.lower()}", model_name)

# -------------------------------------
# Load Data
# -------------------------------------
latent_data = {"Original": {}, "MR": {}}
T_per = {}

for group_name, base_dir in zip(latent_data.keys(), [original_dir, mr_dir]):
    for scenario in scenarios:
        path = os.path.join(base_dir, scenario, "latent", demo_file)
        if os.path.exists(path):
            latent = np.load(path)
            latent_data[group_name][scenario] = latent
            T_per.setdefault(scenario, {})[group_name] = latent.shape[0]
        else:
            latent_data[group_name][scenario] = None
            T_per.setdefault(scenario, {})[group_name] = 0
            print(f"⚠️ Missing: {path}")

# -------------------------------------
# Comparison and CSV Generation
# -------------------------------------
csv_rows = []

for scenario in scenarios:
    data_orig = latent_data["Original"][scenario]
    data_mr = latent_data["MR"][scenario]

    if data_orig is None:
        continue

    T_orig = T_per[scenario]["Original"]
    T_mr = T_per[scenario]["MR"]

    for t in range(min(MAX_T, T_orig)):
        if data_mr is None or t >= T_mr:
            continue

        vec_orig = np.squeeze(data_orig[t])
        vec_mr = np.squeeze(data_mr[t])

        if vec_orig.shape[0] != latent_dim or vec_mr.shape[0] != latent_dim:
            print(f"⚠️ Invalid shape at scenario={scenario}, t={t} → orig={vec_orig.shape}, mr={vec_mr.shape}")
            continue

        top10_orig = np.argsort(vec_orig)[-10:][::-1]
        top10_mr = np.argsort(vec_mr)[-10:][::-1]

        common = np.intersect1d(top10_orig, top10_mr)
        only_orig = np.setdiff1d(top10_orig, top10_mr)
        only_mr = np.setdiff1d(top10_mr, top10_orig)

        if len(common) == 0 and len(only_orig) == 0 and len(only_mr) == 0:
            continue

        activations_orig_common = vec_orig[common]
        activations_mr_common = vec_mr[common]

        activations_orig_only = vec_orig[only_orig]
        activations_mr_only = vec_mr[only_mr]

        csv_rows.append({
            "scenario": scenario,
            "timestep": t,
            "common_neurons": ' '.join(map(str, common)),
            "original_activations_common": ' '.join(f"{v:.6f}" for v in activations_orig_common),
            "mr_activations_common": ' '.join(f"{v:.6f}" for v in activations_mr_common),
            "original_only_neurons": ' '.join(map(str, only_orig)),
            "original_only_activations": ' '.join(f"{v:.6f}" for v in activations_orig_only),
            "mr_only_neurons": ' '.join(map(str, only_mr)),
            "mr_only_activations": ' '.join(f"{v:.6f}" for v in activations_mr_only)
        })

# Save to CSV
df = pd.DataFrame(csv_rows)
output_path = f"top10_neuron_comparison_full_{mr_name.lower()}.csv"
df.to_csv(output_path, index=False)
print(f"✅ Saved detailed comparison CSV: {output_path}")
