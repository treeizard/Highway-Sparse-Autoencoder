import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from tqdm import trange
from util.seed import set_seed

# Config
set_seed()

file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
torch_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
base_model = TorchMLP().to(device)
base_model.load_state_dict(torch.load(torch_model_path, map_location=device))
base_model.eval()

# Register hook to extract 2nd hidden layer activations (ReLU)
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach().cpu())

hook = base_model.net[3].register_forward_hook(hook_fn)

# Collect activations across scenarios
all_activations = []
for scenario in scenarios:
    print(f"Processing scenario: {scenario}")
    data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name, scenario))
    labels_path = os.path.join(data_folder_path, 'labels')
    obs_path = os.path.join(data_folder_path, 'obs')

    # Load data
    X_np, y_np = load_npy_pairs(obs_path, labels_path)
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device).reshape(X_np.shape[0], -1)

    # Forward pass to capture activations
    activations.clear()
    with torch.no_grad():
        _ = base_model(X_tensor)
    layer2_acts = torch.cat(activations, dim=0).to(device)
    all_activations.append(layer2_acts)

hook.remove()

# Merge all activations
X_all = torch.cat(all_activations, dim=0)  # Shape [total_N, 256]

# Train Sparse Autoencoder
sae = SparseAutoencoder(input_dim=256, hidden_dim=1024, sparsity_lambda=5e-3).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir='runs/super_dqn_sae')

n_epochs = 40000
for epoch in trange(n_epochs, desc="Training SAE"):
    sae.train()
    x_hat, z = sae(X_all)
    loss = sae.loss(X_all, x_hat, z)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("Loss/train", loss.item(), epoch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

writer.close()

# Save SAE model
ckpt_dir = os.path.join(os.path.dirname(__file__), 'sae_ckpt')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, 'sae_super_dqn.pt')
torch.save(sae.state_dict(), ckpt_path)
print(f"SAE model saved to {ckpt_path}")

# Extract sparse codes
sae.eval()
with torch.no_grad():
    _, sparse_codes = sae(X_all)

# Top-10 active neurons
topk_vals, topk_inds = sparse_codes.topk(k=10, dim=1)
all_top_indices = topk_inds.flatten().cpu().numpy()
unique, counts = np.unique(all_top_indices, return_counts=True)
sorted_pairs = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
top10_indices = [idx for idx, _ in sorted_pairs[:10]]
top10_activations = sparse_codes[:, top10_indices].detach().cpu().numpy()

# Plot overall histogram
plt.figure(figsize=(10, 6))
for i, idx in enumerate(top10_indices):
    plt.hist(top10_activations[:, i], bins=50, alpha=0.6, label=f"Neuron {idx}")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.title("Histogram of Top 10 Activated Neuron Units")
plt.legend()
plt.tight_layout()
plt.show()

# Subplot per neuron
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 12))
axes = axes.flatten()
for i, idx in enumerate(top10_indices):
    axes[i].hist(top10_activations[:, i], bins=50, color='skyblue', edgecolor='black')
    axes[i].set_title(f"Neuron {idx}")
    axes[i].set_xlabel("Activation Value")
    axes[i].set_ylabel("Frequency")
for j in range(len(top10_indices), len(axes)):
    axes[j].axis('off')

plt.suptitle("Activation Histograms of Top 10 Latent Units", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
