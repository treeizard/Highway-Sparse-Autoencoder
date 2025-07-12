import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ------------------ Load Data ------------------
data = np.load('recorded_latents.npz')
inputs = data['inputs']   # shape [N, 25]
latents = data['latents'] # shape [N, 1024]

# ------------------ Choose Subset ------------------
N = min(3000, len(inputs))  # for clarity
inputs = inputs[:N]
latents = latents[:N]

# ------------------ Preprocess ------------------
# Optional: normalize input and latent
inputs_norm = (inputs - inputs.mean(0)) / (inputs.std(0) + 1e-6)
latents_norm = (latents - latents.mean(0)) / (latents.std(0) + 1e-6)

# ------------------ Projection ------------------
print("Running PCA on input...")
input_pca = PCA(n_components=2).fit_transform(inputs_norm)

print("Running t-SNE on latent...")
latent_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', max_iter=1000, random_state=42).fit_transform(latents_norm)

# ------------------ Plot ------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(input_pca[:, 0], input_pca[:, 1], s=8, alpha=0.6, c='dodgerblue')
axes[0].set_title("PCA Projection of 25-D Inputs")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].grid(True)

axes[1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], s=8, alpha=0.6, c='crimson')
axes[1].set_title("t-SNE of 1024-D Latent Codes")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")
axes[1].grid(True)

plt.suptitle("Visualizing Input vs Latent Representations")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
