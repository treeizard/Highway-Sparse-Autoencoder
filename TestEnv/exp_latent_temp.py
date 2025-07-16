import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------
# Config
# -------------------------------------
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo1.npy'
latent_dim = 1024
model_name = "super_dqn_sae"
latent_side = int(np.sqrt(latent_dim))

original_dir = os.path.join("data_original", model_name)
mr_dir = os.path.join("data_dyna_test", "data_mr23", model_name)  # Change MR here

# -------------------------------------
# Load data
# -------------------------------------
latent_data = {
    "Original": {},
    "MR": {}
}
T_per = {}

for group_name, base_dir in zip(latent_data.keys(), [original_dir, mr_dir]):
    for scenario in scenarios:
        path = os.path.join(base_dir, scenario, "latent", demo_file)
        if os.path.exists(path):
            latent = np.load(path)  # shape [T, 1024]
            latent_data[group_name][scenario] = latent
            T_per[scenario] = latent.shape[0]
        else:
            print(f"Missing file: {path}")
            latent_data[group_name][scenario] = None
            T_per[scenario] = 0

# -------------------------------------
# Visualization
# -------------------------------------
fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 9))
plt.subplots_adjust(bottom=0.15, hspace=0.4)
images = []
texts = []

vmin = min(np.min(v) for g in latent_data.values() for v in g.values() if v is not None)
vmax = max(np.max(v) for g in latent_data.values() for v in g.values() if v is not None)

for i, scenario in enumerate(scenarios):
    for j, group in enumerate(["Original", "MR"]):
        ax = axes[i][j]
        ax.set_title(f"{group} - {scenario}")
        ax.axis("off")

        data = latent_data[group][scenario]
        if data is not None and T_per[scenario] > 0:
            latent_2d = data[0].reshape(latent_side, latent_side)
            im = ax.imshow(latent_2d, cmap='viridis', vmin=vmin, vmax=vmax)
            txt = ax.text(0.01, 0.99, "", ha='left', va='top', transform=ax.transAxes, fontsize=9, color='white')
        else:
            im = ax.imshow(np.zeros((latent_side, latent_side)), cmap='viridis')
            txt = ax.text(0.01, 0.99, "N/A", ha='left', va='top', transform=ax.transAxes, fontsize=9, color='white')

        images.append(im)
        texts.append(txt)

# -------------------------------------
# Slider
# -------------------------------------
slider_ax = plt.axes([0.3, 0.03, 0.4, 0.03])
slider = Slider(slider_ax, 'Timestep', 0, max(T_per.values()) - 1, valinit=0, valstep=1)

# -------------------------------------
# Update Function
# -------------------------------------
def update(t):
    t = int(t)
    for i, scenario in enumerate(scenarios):
        for j, group in enumerate(["Original", "MR"]):
            idx = 2 * i + j
            data = latent_data[group][scenario]
            if data is not None and t < data.shape[0]:
                latent_flat = data[t]
                images[idx].set_data(latent_flat.reshape(latent_side, latent_side))
                top10_idx = np.argsort(latent_flat)[-10:][::-1]
                texts[idx].set_text("Top-10:\n" + "\n".join(str(k) for k in top10_idx))
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.suptitle("Top 10 Neuron Activation Comparison\n(Original vs MR23)", fontsize=14)
plt.show()
