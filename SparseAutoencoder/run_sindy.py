import os
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from tqdm import tqdm

# ------------------ Load Data ------------------
data = np.load('data_processed/recorded_latents.npz')
X = data['inputs']      # shape [N, 25]
Z = data['latents']     # shape [N, 1024]

n_latents = Z.shape[1]

# ------------------ SINDy Config ------------------
library = ps.PolynomialLibrary(degree=1, include_bias=True, include_interaction=True)
optimizer = ps.STLSQ(threshold=1e-2)

models = []
equations = []
losses = []

# ------------------ Fit All Latents ------------------
print("📐 Fitting polynomial approximations for each latent unit...")
for i in tqdm(range(n_latents), desc="Fitting SINDy"):
    z_i = Z[:, i].reshape(-1, 1)
    model = ps.SINDy(feature_library=library, optimizer=optimizer, discrete_time=True)
    model.fit(X, x_dot=z_i)

    z_pred = model.predict(X)
    mse = np.mean((z_pred - z_i) ** 2)
    norm_mse = mse / (np.var(z_i) + 1e-8)  # normalized by variance
    losses.append(norm_mse)

    models.append(model)
    equations.append(model.equations()[0])

# ------------------ Save Equations ------------------
os.makedirs("data_processed", exist_ok=True)
eq_path = os.path.join("data_processed", "sindy_latent_equations.txt")
with open(eq_path, "w") as f:
    for i, eq in enumerate(equations):
        f.write(f"z_{i} = {eq}\n")

print(f"✅ Saved all 1024 symbolic equations to {eq_path}")

# ------------------ Save Good-Fit Equations ------------------
good_eq_path = os.path.join("data_processed", "sindy_latent_equations_good_fit.txt")
with open(good_eq_path, "w") as f:
    for i, (eq, loss_val) in enumerate(zip(equations, losses)):
        if loss_val < 0.2:
            f.write(f"z_{i} = {eq}\n")

print(f"✅ Saved symbolic equations with normalized loss < 0.2 to {good_eq_path}")


# ------------------ Save Loss Histogram ------------------
plt.figure(figsize=(10, 5))
plt.hist(losses, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Normalized MSE Loss")
plt.ylabel("Number of Latents")
plt.title("Distribution of Normalized SINDy Approximation Loss")
plt.tight_layout()

loss_hist_path = os.path.join("data_processed", "sindy_fit_loss_histogram.png")
plt.savefig(loss_hist_path)
plt.close()
print(f"📊 Saved histogram to {loss_hist_path}")

# ------------------ Save 32x32 Heatmap ------------------
losses = np.array(losses)
assert losses.shape[0] == 1024, "Expected 1024 latents"
loss_grid = losses.reshape(32, 32)

plt.figure(figsize=(8, 6))
im = plt.imshow(loss_grid, cmap='viridis', interpolation='nearest')
plt.title("SINDy Normalized Loss Heatmap (1024 Latents)")
plt.colorbar(im, label="Normalized MSE Loss")
plt.xticks([])
plt.yticks([])
plt.tight_layout()

heatmap_path = os.path.join("data_processed", "sindy_loss_heatmap.png")
plt.savefig(heatmap_path)
plt.close()
print(f"🟦 Saved heatmap to {heatmap_path}")

# ------------------ Save Masked Good-Fit Heatmap ------------------
good_mask = losses < 0.2
masked_grid = np.full_like(losses, fill_value=np.nan)
masked_grid[good_mask] = losses[good_mask]
masked_grid = masked_grid.reshape(32, 32)

plt.figure(figsize=(8, 6))
im = plt.imshow(masked_grid, cmap='viridis', interpolation='nearest')
plt.title("Good-Fit Latents (Normalized Loss < 0.2)")
cbar = plt.colorbar(im, label="Normalized MSE Loss")
plt.xticks([])
plt.yticks([])
plt.tight_layout()

good_heatmap_path = os.path.join("data_processed", "sindy_loss_heatmap_good_fit.png")
plt.savefig(good_heatmap_path)
plt.close()
print(f"🟩 Saved good-fit heatmap to {good_heatmap_path}")
