import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs
from tqdm import trange

# Config
file_name = 'highway_dqn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name))
labels_path = os.path.join(data_folder_path, 'labels')
obs_path = os.path.join(data_folder_path, 'obs')



# Load observations and labels
X_np, y_np = load_npy_pairs(obs_path, labels_path)
X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
X_tensor = X_tensor.reshape(X_tensor.shape[0], -1)

# Load Torch model
torch_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))

# Instantiate and load state_dict
base_model = TorchMLP().to(device)
base_model.load_state_dict(torch.load(torch_model_path, map_location=device))
base_model.eval()

# Register hook to extract activations from second ReLU
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach().cpu())

hook = base_model.net[3].register_forward_hook(hook_fn)

# Forward pass to collect activations
with torch.no_grad():
    _ = base_model(X_tensor)

hook.remove()
layer2_acts = torch.cat(activations, dim=0).to(device)  # Shape: [N, 256]

# Define sparse autoencoder


# Train SAE
sae = SparseAutoencoder(input_dim=256).to(device)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

# Setup TensorBoard writer
log_dir = 'runs'
writer = SummaryWriter(log_dir=log_dir)

n_epochs = 20000  # You can adjust this
for epoch in trange(n_epochs, desc="Training SAE"):
    sae.train()
    x_hat, z = sae(layer2_acts)
    loss = sae.loss(layer2_acts, x_hat, z)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log to TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Optional: print some checkpoints
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

writer.close()

# Extract sparse codes
sae.eval()
with torch.no_grad():
    _, sparse_codes = sae(layer2_acts)

# Top-k indices per input
topk_vals, topk_inds = sparse_codes.topk(k=10, dim=1)

# 1. Count top latent activations
all_top_indices = topk_inds.flatten().cpu().numpy()  # shape: [N * 10]

# 2. Compute histogram over all indices
unique, counts = np.unique(all_top_indices, return_counts=True)
sorted_pairs = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
top10_indices = [idx for idx, _ in sorted_pairs[:10]]

# 3. Gather activation values for those top 10 units
top10_activations = sparse_codes[:, top10_indices].detach().cpu().numpy()  # shape: [N, 10]

# 4. Plot histogram
plt.figure(figsize=(10, 6))
for i, idx in enumerate(top10_indices):
    plt.hist(top10_activations[:, i], bins=50, alpha=0.6, label=f"Neuron {idx}")

plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.title("Histogram of Top 10 Activated Neuron Units")
plt.legend()
plt.tight_layout()
plt.show()


# Plot each top latent activation as a separate subplot
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 12))
axes = axes.flatten()

for i, idx in enumerate(top10_indices):
    axes[i].hist(top10_activations[:, i], bins=50, color='skyblue', edgecolor='black')
    axes[i].set_title(f"Neuron {idx}")
    axes[i].set_xlabel("Activation Value")
    axes[i].set_ylabel("Frequency")

# Hide any unused axes
for j in range(len(top10_indices), len(axes)):
    axes[j].axis('off')

plt.suptitle("Activation Histograms of Top 10 Latent Units", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
