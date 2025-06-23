import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch.nn.functional as F
from tqdm import tqdm  # <-- progress bar

from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.seed import set_seed

# ------------------ Config ------------------
set_seed()
file_name = 'super_dqn'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 1024
input_dim = 25  # 5 vehicles × 5 features

# ------------------ Load Models ------------------
base_model = TorchMLP().to(device)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'torch_model', file_name + '.pt'))
base_model.load_state_dict(torch.load(model_path, map_location=device))
base_model.eval()

sae = SparseAutoencoder(input_dim=256, hidden_dim=hidden_dim, sparsity_lambda=5e-3).to(device)
sae_path = os.path.join(os.getcwd(), 'sae_ckpt', 'sae_super_dqn.pt')
sae.load_state_dict(torch.load(sae_path, map_location=device))
sae.eval()

# ------------------ DeepDream Optimizer ------------------
def dream_input_for_neuron(model, sae, neuron_idx, trials=10, steps=300, lr=1e-2, reg_lambda=1e-3):
    best_x = None
    best_activation = -float('inf')

    for _ in range(trials):
        x = torch.randn(1, input_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            h = model.net[0](x)
            h = F.relu(h)
            z = sae.encoder(h)
            act = z[:, neuron_idx].mean()
            loss = -act + reg_lambda * (x ** 2).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_final = F.relu(model.net[0](x))
            z_final = sae.encoder(h_final)
            final_act = z_final[:, neuron_idx].item()

        if final_act > best_activation:
            best_activation = final_act
            best_x = x.detach().clone()

    return best_x.cpu().numpy().squeeze(), best_activation

# ------------------ Precompute All Dream Inputs ------------------
print("Precomputing best dream inputs for all neurons...")
dream_cache = {}
for idx in tqdm(range(hidden_dim), desc="Optimizing Neurons", ncols=100):
    vec, act = dream_input_for_neuron(base_model, sae, idx)
    dream_cache[idx] = (vec, act)
print("All neuron dreams cached.\n")

# ------------------ Set Up Plot ------------------
fig = plt.figure(figsize=(8, 6))
ax_main = fig.add_axes([0.1, 0.35, 0.8, 0.6])
ax_slider = fig.add_axes([0.25, 0.1, 0.5, 0.03])

# ------------------ Plot Function ------------------
def plot_neuron(neuron_idx):
    ax_main.cla()
    vec, act = dream_cache[neuron_idx]

    for i in range(5):
        base = i * 5
        x = vec[base + 1]
        y = vec[base + 2]
        u = vec[base + 3]
        v = vec[base + 4]

        color = 'red' if i == 0 else 'blue'
        label = 'Ego' if i == 0 else f'V{i}'
        ax_main.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=color, alpha=0.6)
        ax_main.scatter(x, y, s=80, c=color, label=label)

    ax_main.set_xlim(-5, 5)
    ax_main.set_ylim(-5, 5)
    ax_main.set_title(f"Neuron {neuron_idx} — Max Activation: {act:.3f}")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.grid(True)
    ax_main.legend(loc='upper right', fontsize=8)
    fig.canvas.draw_idle()

# ------------------ Slider Control ------------------
slider = Slider(ax_slider, 'Neuron Index', 0, hidden_dim - 1, valinit=0, valstep=1)
slider.on_changed(lambda val: plot_neuron(int(round(val))))

# ------------------ Initial Display ------------------
plot_neuron(0)
plt.show()
