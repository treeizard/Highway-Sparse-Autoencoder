import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder

# ------------------ Config ------------------
def set_seed(seed=45):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neuron_index = 894
perturb_factors = np.linspace(0.25, 1.5, 5)

# ------------------ Load Models ------------------
base_model = TorchMLP().to(device)
model_path = os.path.join("..", "BaseModel", "torch_model", file_name + ".pt")
base_model.load_state_dict(torch.load(model_path, map_location=device))
base_model.eval()

sae = SparseAutoencoder(input_dim=256, hidden_dim=1024, sparsity_lambda=5e-3).to(device)
sae_ckpt_path = os.path.join("sae_ckpt", "sae_super_dqn_best.pt")
sae.load_state_dict(torch.load(sae_ckpt_path, map_location=device))
sae.eval()

# ------------------ Collect 5 Demos per Scenario ------------------
demo_samples = []

for scenario in scenarios:
    obs_dir = os.path.join("..", "DataEngine", "data", file_name, scenario, "obs")
    label_dir = os.path.join("..", "DataEngine", "data", file_name, scenario, "labels")
    if not os.path.exists(obs_dir):
        continue
    demo_files = [f for f in os.listdir(obs_dir) if f.endswith(".npy")]
    sampled = random.sample(demo_files, min(5, len(demo_files)))
    for f in sampled:
        demo_samples.append({
            "scenario": scenario,
            "obs_path": os.path.join(obs_dir, f),
            "label_path": os.path.join(label_dir, f),
            "demo_id": f
        })

# ------------------ Plotting ------------------
fig, axes = plt.subplots(3, 5, figsize=(22, 12), sharex=False, sharey=True)
axes = axes.flatten()

for i, demo in enumerate(demo_samples):
    scenario = demo["scenario"]
    obs_path = demo["obs_path"]
    label_path = demo["label_path"]
    demo_id = demo["demo_id"]

    X_np = np.load(obs_path)
    y_np = np.load(label_path)

    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device).reshape(X_np.shape[0], -1)
    Y_tensor = torch.tensor(y_np, dtype=torch.long).cpu().numpy()

    with torch.no_grad():
        h1 = base_model.net[:2](X_tensor)
        h2 = base_model.net[2:4](h1)
        latent = sae.relu(sae.encoder(h2))
        base_recon = sae.decoder(latent)
        base_pred = torch.argmax(base_model.net[4:](base_recon), dim=1).cpu().numpy()

    ax = axes[i]
    ax.plot(Y_tensor, label="Ground Truth", linestyle=":", linewidth=1)
    ax.plot(base_pred, label="Original", linestyle="--", linewidth=1)

    for factor in perturb_factors:
        latent_mod = latent.clone()
        latent_mod[:, neuron_index] += factor
        with torch.no_grad():
            recon_mod = sae.decoder(latent_mod)
            pred_mod = torch.argmax(base_model.net[4:](recon_mod), dim=1).cpu().numpy()
        ax.plot(pred_mod, label=f"+{factor:.1f}", alpha=0.4, linewidth=1)

    ax.set_title(f"{scenario} | {demo_id}", fontsize=9)
    ax.grid(True)

    if i == 0:
        ax.legend(fontsize="x-small", loc="upper right")

# Hide any unused subplots (not expected here, but safe)
for j in range(len(demo_samples), len(axes)):
    axes[j].axis('off')

fig.suptitle(f"Neuron {neuron_index} Perturbation across Scenarios", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
