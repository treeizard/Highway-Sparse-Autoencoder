import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from util.seed import set_seed

# ------------------ Config ------------------
set_seed()
file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 1024  # Must be square (e.g., 1024 = 32x32)
side_len = int(hidden_dim ** 0.5)

# ------------------ Load Models ------------------
base_model = TorchMLP().to(device)
base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
base_model.load_state_dict(torch.load(base_model_path, map_location=device))
base_model.eval()

sae = SparseAutoencoder(input_dim=256, hidden_dim=hidden_dim, sparsity_lambda=5e-3).to(device)
sae_ckpt_path = os.path.join(os.path.dirname(__file__), 'sae_ckpt', 'sae_super_dqn_best.pt')
sae.load_state_dict(torch.load(sae_ckpt_path, map_location=device))
sae.eval()

# ------------------ Hook for Activations ------------------
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach().cpu())
base_model.net[3].register_forward_hook(hook_fn)

# ------------------ Storage for RGB channels ------------------
rgb_layers = []

# ------------------ Process Each Scenario ------------------
for scenario in scenarios:
    print(f"Processing {scenario}...")

    # Load data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name, scenario))
    obs_path = os.path.join(data_dir, 'obs')
    labels_path = os.path.join(data_dir, 'labels')
    X_np, _ = load_npy_pairs(obs_path, labels_path)
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device).reshape(X_np.shape[0], -1)

    # Extract hidden layer activations
    activations.clear()
    with torch.no_grad():
        _ = base_model(X_tensor)
    hidden_acts = torch.cat(activations, dim=0).to(device)

    # SAE latent codes
    with torch.no_grad():
        _, sparse_codes = sae(hidden_acts)

    # Binary activation proportion
    binary_mask = (sparse_codes > 0).float()
    activation_rate = binary_mask.mean(dim=0).cpu().numpy().reshape(side_len, side_len)

    # Clip to [0, 1] and append
    rgb_layers.append(np.clip(activation_rate, 0.0, 1.0))

# ------------------ Compose RGB Image ------------------
rgb_image = np.stack(rgb_layers, axis=-1)  # shape: (H, W, 3)

# ------------------ Plotting ------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

titles = ['Highway (Red)', 'Merge (Green)', 'Roundabout (Blue)', 'Combined (RGB)']
cmap_channels = ['Reds', 'Greens', 'Blues']

# Individual channels
for i in range(3):
    ax = axes[i // 2][i % 2]
    ax.imshow(rgb_layers[i], cmap=cmap_channels[i], vmin=0, vmax=1)
    ax.set_title(titles[i], fontsize=12)
    ax.axis('off')

# Combined RGB image
ax = axes[1][1]
ax.imshow(rgb_image)
ax.set_title(titles[3], fontsize=12)
ax.axis('off')

plt.suptitle("Latent Neuron Activation Maps by Scenario (RGB Overlay)", fontsize=16)
plt.tight_layout()
plt.show()
