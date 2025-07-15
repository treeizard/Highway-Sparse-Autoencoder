import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------------------------
# Config
# -------------------------------------
scenarios = ['highway', 'merge', 'roundabout']
demo_file = 'demo2.npy'
latent_dim = 1024
latent_side = int(np.sqrt(latent_dim))  # 32x32
model_name = "super_dqn_sae"

base_dirs = {
    "Original": os.path.join("data_original", model_name),
    "MR23": os.path.join("data_dyna_test", "data_mr23", model_name),
    "MR24": os.path.join("data_dyna_test", "data_mr24", model_name),
    "MR34": os.path.join("data_dyna_test", "data_mr34", model_name)
}

# -------------------------------------
# Load all latent data
# -------------------------------------
latent_data = {}
T_per = {}

for group_name, base_dir in base_dirs.items():
    latent_data[group_name] = {}
    T_per[group_name] = {}
    for scenario in scenarios:
        path = os.path.join(base_dir, scenario, "latent", demo_file)
        if os.path.exists(path):
            latent = np.load(path)  # shape: [T, 1024]
            latent_data[group_name][scenario] = latent
            T_per[group_name][scenario] = latent.shape[0]
        else:
            print(f"Warning: Missing file {path}")
            latent_data[group_name][scenario] = None
            T_per[group_name][scenario] = 0

# -------------------------------------
# Setup plot grid and sliders
# -------------------------------------
fig, axes = plt.subplots(len(base_dirs), len(scenarios), figsize=(6 * len(scenarios), 3.5 * len(base_dirs)))
plt.subplots_adjust(bottom=0.15)
images = []
sliders = []
slider_axes = []

# Global vmin/vmax for color scaling
vmin = min(np.min(v) for group in latent_data.values() for v in group.values() if v is not None)
vmax = max(np.max(v) for group in latent_data.values() for v in group.values() if v is not None)

# Loop over rows and columns
for row, group_name in enumerate(base_dirs.keys()):
    for col, scenario in enumerate(scenarios):
        ax = axes[row][col] if len(base_dirs) > 1 else axes[col]
        ax.set_title(f"{group_name} - {scenario}")
        ax.axis("off")

        data = latent_data[group_name][scenario]
        T = T_per[group_name][scenario]

        if data is not None and T > 0:
            latent_2d = data[0].reshape(latent_side, latent_side)
            im = ax.imshow(latent_2d, cmap='viridis', vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(np.zeros((latent_side, latent_side)), cmap='viridis')
        images.append(im)

        # Slider
        slider_x = 0.15 + col * 0.26
        slider_y = 0.08 - 0.05 * row
        sa = plt.axes([slider_x, max(0.01, slider_y), 0.22, 0.03])
        slider = Slider(
            ax=sa,
            label=f"{group_name}-{scenario}",
            valmin=0,
            valmax=max(0, T - 1),
            valinit=0,
            valstep=1
        )
        sliders.append(slider)
        slider_axes.append(sa)

fig.suptitle(f"Latent Heatmaps | Demo: {demo_file}", fontsize=16)

# -------------------------------------
# Update functions
# -------------------------------------
def make_update_func(index, group_name, scenario):
    def update(val):
        t = int(sliders[index].val)
        data = latent_data[group_name][scenario]
        if data is not None and t < data.shape[0]:
            images[index].set_data(data[t].reshape(latent_side, latent_side))
            fig.canvas.draw_idle()
    return update

# Register all sliders
index = 0
for row, group_name in enumerate(base_dirs.keys()):
    for col, scenario in enumerate(scenarios):
        sliders[index].on_changed(make_update_func(index, group_name, scenario))
        index += 1

plt.show()