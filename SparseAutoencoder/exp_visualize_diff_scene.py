import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from util.seed import set_seed

# Config
set_seed()
file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained base model
torch_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
base_model = TorchMLP().to(device)
base_model.load_state_dict(torch.load(torch_model_path, map_location=device))
base_model.eval()

# Load trained SAE model
sae = SparseAutoencoder(input_dim=256, hidden_dim=1024, sparsity_lambda=5e-3).to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), 'sae_ckpt', 'sae_super_dqn.pt')
sae.load_state_dict(torch.load(ckpt_path, map_location=device))
sae.eval()

# Hook for activation extraction
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach().cpu())

base_model.net[3].register_forward_hook(hook_fn)

# Plotting setup
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 15))
colors = ['skyblue', 'salmon', 'lightgreen']

for idx, scenario in enumerate(scenarios):
    print(f"Processing scenario: {scenario}")
    data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name, scenario))
    labels_path = os.path.join(data_folder_path, 'labels')
    obs_path = os.path.join(data_folder_path, 'obs')

    # Load data
    X_np, y_np = load_npy_pairs(obs_path, labels_path)
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device).reshape(X_np.shape[0], -1)

    # Forward to get activations
    activations.clear()
    with torch.no_grad():
        _ = base_model(X_tensor)
    layer2_acts = torch.cat(activations, dim=0).to(device)

    # Pass through SAE
    with torch.no_grad():
        _, sparse_codes = sae(layer2_acts)

    # Top-10 indices for this scenario
    topk_vals, topk_inds = sparse_codes.topk(k=10, dim=1)
    all_top_indices = topk_inds.flatten().cpu().numpy()
    unique, counts = np.unique(all_top_indices, return_counts=True)
    sorted_pairs = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    top10_indices = [idx for idx, _ in sorted_pairs[:10]]
    top10_activations = sparse_codes[:, top10_indices].detach().cpu().numpy()

    # Plot histograms
    ax = axes[idx]
    for i in range(10):
        ax.hist(top10_activations[:, i], bins=40, alpha=0.5, label=f"Neuron {top10_indices[i]}")
    ax.set_title(f"{scenario.capitalize()} Top-10 Latents")
    ax.set_xlabel("Activation")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7)

plt.suptitle("Top-10 Activated SAE Neurons per Scenario", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
