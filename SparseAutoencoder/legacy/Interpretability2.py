import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import torch.nn.functional as F

# Config
file_name = 'highway_dqn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataEngine', 'data', file_name))
labels_path = os.path.join(data_folder_path, 'labels')
obs_path = os.path.join(data_folder_path, 'obs')

# Load data
def load_npy_pairs(obs_dir, label_dir):
    obs_files = sorted([f for f in os.listdir(obs_dir) if f.endswith('.npy')])
    X, y = [], []
    for fname in obs_files:
        obs = np.load(os.path.join(obs_dir, fname))
        label = np.load(os.path.join(label_dir, fname))
        X.append(obs)
        y.append(label)
    return np.vstack(X), np.concatenate(y)

# Load observations and labels
X_np, y_np = load_npy_pairs(obs_path, labels_path)

X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
X_tensor = X_tensor.reshape(X_tensor.shape[0], -1)
# Load Torch model
torch_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        q_values = self.net(x)
        return torch.argmax(q_values, dim=1)

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


def kl_divergence_diag(mu, logvar, mu0, logvar0):
    # logvar and logvar0 are logs of variance (not std)
    var = torch.exp(logvar)
    var0 = torch.exp(logvar0)

    kl = 0.5 * torch.sum(
        logvar0 - logvar + (var + (mu - mu0).pow(2)) / var0 - 1,
        dim=1
    )
    return kl.mean()

class SparseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, sparsity_lambda=0.05,
                 prior_mu=0.0, prior_logvar=0.0):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        self.relu = nn.ReLU()
        self.sparsity_lambda = sparsity_lambda

        # Register buffers for prior parameters
        self.register_buffer("prior_mu", torch.full((latent_dim,), prior_mu))
        self.register_buffer("prior_logvar", torch.full((latent_dim,), prior_logvar))

    def encode(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def loss(self, x, x_hat, mu, logvar, z):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kl_div = kl_divergence_diag(mu, logvar, self.prior_mu, self.prior_logvar)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        return recon_loss + kl_div + sparsity_loss


# Train SAE
vae = SparseVAE(input_dim=256, latent_dim = 2304).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Setup TensorBoard writer
log_dir = 'runs'
writer = SummaryWriter(log_dir=log_dir)

n_epochs = 20000  # You can adjust this
for epoch in trange(n_epochs, desc="Training Sparse VAE"):
    vae.train()
    x_hat, mu, logvar, z = vae(layer2_acts)
    loss = vae.loss(layer2_acts, x_hat, mu, logvar, z)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss.item(), epoch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

writer.close()

# Extract sparse codes
vae.eval()
with torch.no_grad():
     _, _, _, sparse_codes = vae(layer2_acts)

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


# Compute mean and std of latent dimensions
latent_means = sparse_codes.mean(dim=0).cpu().numpy()
latent_stds = sparse_codes.std(dim=0).cpu().numpy()


# Determine grid size for square matrix plot
latent_dim = sparse_codes.shape[1]
side_len = math.ceil(math.sqrt(latent_dim))
print(latent_dim)

def plot_latent_heatmap(data, title, cmap="coolwarm"):
    matrix = np.zeros((side_len, side_len))
    matrix.flat[:latent_dim] = data  # Fill matrix (pad remaining with 0)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Plot mean activation heatmap
plot_latent_heatmap(latent_means, "Mean Latent Activations")

# Plot std deviation heatmap
plot_latent_heatmap(latent_stds, "Std Dev of Latent Activations")