import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import pysindy as ps
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from util.seed import set_seed

# ------------------ Config ------------------
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
hidden_dim = 1024
side_len = int(hidden_dim ** 0.5)

# ------------------ Load Latent Data for SINDy ------------------
data = np.load('data_processed/recorded_latents.npz')
X = data['inputs']      # shape [N, 25]
Z = data['latents']     # shape [N, 1024]
n_latents = Z.shape[1]

# ------------------ SINDy Setup ------------------
library = ps.PolynomialLibrary(degree=1, include_bias=True, include_interaction=True)
optimizer = ps.STLSQ(threshold=5e-2)

losses = []
equations = []

print("📐 Fitting SINDy models for 1024 latent units...")
for i in tqdm(range(n_latents), desc="Fitting SINDy"):
    z_i = Z[:, i].reshape(-1, 1)
    model = ps.SINDy(feature_library=library, optimizer=optimizer, discrete_time=True)
    model.fit(X, x_dot=z_i)
    z_pred = model.predict(X)
    mse = np.mean((z_pred - z_i) ** 2)
    norm_mse = mse / (np.var(z_i) + 1e-8)
    losses.append(norm_mse)
    equations.append(model.equations()[0])

losses = np.array(losses)
good_fit_indices = np.where(losses < 0.2)[0]

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

# ------------------ Process Scenarios and Gather Activations ------------------
rgb_layers = []
for scenario in scenarios:
    print(f"🧭 Processing scenario: {scenario}")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name, scenario))
    X_np, _ = load_npy_pairs(os.path.join(data_dir, 'obs'), os.path.join(data_dir, 'labels'))
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device).reshape(X_np.shape[0], -1)

    activations.clear()
    with torch.no_grad():
        _ = base_model(X_tensor)
    hidden_acts = torch.cat(activations, dim=0).to(device)

    with torch.no_grad():
        _, sparse_codes = sae(hidden_acts)

    binary_mask = (sparse_codes > 0).float()
    activation_rate = binary_mask.mean(dim=0).cpu().numpy().reshape(side_len, side_len)
    rgb_layers.append(np.clip(activation_rate, 0.0, 1.0))

# ------------------ Compute Average Activation Map ------------------
avg_activation_map = np.mean(np.stack(rgb_layers, axis=0), axis=0)
avg_activation_flat = avg_activation_map.flatten()
top20_indices = np.argsort(avg_activation_flat)[-20:][::-1]

# ------------------ Compare Top-20 with Good-Fit SINDy ------------------
intersection = np.intersect1d(top20_indices, good_fit_indices)
print(f"🔍 Top-20 most active latent indices: {top20_indices}")
print(f"✅ Good-fit SINDy latent indices (loss < 0.2): {good_fit_indices}")
print(f"🧩 Intersection (good-fit & top-20 active): {intersection}")

# ------------------ Plot Intersection Mask ------------------
intersection_mask = np.zeros_like(avg_activation_flat)
intersection_mask[intersection] = 1
intersection_mask = intersection_mask.reshape(32, 32)

plt.figure(figsize=(8, 6))
plt.imshow(intersection_mask, cmap='coolwarm')
plt.title("Overlap: Top-20 Active & Good-Fit SINDy Latents")
plt.colorbar(label="1 = Overlap")
plt.tight_layout()
plt.savefig("data_processed/top20_goodfit_intersection.png")
plt.close()
print("📌 Saved intersection visualization to data_processed/top20_goodfit_intersection.png")
