import os
import torch
import numpy as np
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from util.seed import set_seed

# ------------------ Config ------------------
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']

# ------------------ Load Base Model ------------------
base_model = TorchMLP().to(device)
base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
base_model.load_state_dict(torch.load(base_model_path, map_location=device))
base_model.eval()

# ------------------ Load SAE ------------------
sae = SparseAutoencoder(input_dim=256, hidden_dim=1024, sparsity_lambda=5e-3).to(device)
sae_ckpt_path = os.path.join(os.getcwd(), 'sae_ckpt', 'sae_super_dqn_best.pt')
sae.load_state_dict(torch.load(sae_ckpt_path, map_location=device))
sae.eval()

# ------------------ Collect Input + Latent ------------------
all_inputs, all_latents = [], []

for scenario in scenarios:
    print(f"Processing scenario: {scenario}")
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name, scenario))
    obs_path = os.path.join(data_folder, 'obs')
    labels_path = os.path.join(data_folder, 'labels')

    X_np, _ = load_npy_pairs(obs_path, labels_path)
    X_flat = torch.tensor(X_np, dtype=torch.float32).reshape(X_np.shape[0], -1).to(device)

    # Feed to base model → extract 256-dim features
    with torch.no_grad():
        base_hidden = torch.relu(base_model.net[0](X_flat))  # assumes 1st layer to be correct pre-sae input
        z = sae.encoder(base_hidden)

    all_inputs.append(X_flat.cpu().numpy())
    all_latents.append(z.cpu().numpy())

# Stack and Save -----------------------------------
input_array = np.vstack(all_inputs)     # shape [N, 25]
latent_array = np.vstack(all_latents)   # shape [N, 1024]

save_path = os.path.join(os.getcwd(), 'data_processed/recorded_latents.npz')
np.savez_compressed(save_path, inputs=input_array, latents=latent_array)
print(f"Saved input-latent pairs to {save_path}")
