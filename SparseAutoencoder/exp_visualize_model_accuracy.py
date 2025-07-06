import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.base_model import TorchMLP
from models.SAE import SparseAutoencoder
from util.data_tool import load_npy_pairs

# ------------------ Config ------------------
def set_seed(seed=0):
    torch.manual_seed(seed)
set_seed()

file_name = 'super_dqn'
scenarios = ['highway', 'merge', 'roundabout']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load Base Model ------------------
base_model = TorchMLP().to(device)
model_path = os.path.join("..", "BaseModel", "torch_model", file_name + ".pt")
base_model.load_state_dict(torch.load(model_path, map_location=device))
base_model.eval()

# ------------------ Load SAE ------------------
sae = SparseAutoencoder(input_dim=256, hidden_dim=1024, sparsity_lambda=5e-3).to(device)
sae_ckpt_path = os.path.join("sae_ckpt", "sae_super_dqn_best.pt")
sae.load_state_dict(torch.load(sae_ckpt_path, map_location=device))
sae.eval()

# ------------------ Evaluation Loop ------------------
scenario_metrics = []
all_activations = []
all_labels = []

for scenario in scenarios:
    print(f"\n📥 Processing scenario: {scenario}")
    obs_dir = os.path.join("..", "DataEngine", "data", file_name, scenario, "obs")
    label_dir = os.path.join("..", "DataEngine", "data", file_name, scenario, "labels")

    X_np, y_np = load_npy_pairs(obs_dir, label_dir)
    X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device).reshape(X_np.shape[0], -1)
    Y_tensor = torch.tensor(y_np, dtype=torch.long, device=device)

    with torch.no_grad():
        h1 = base_model.net[:2](X_tensor)
        h2 = base_model.net[2:4](h1)
        base_logits = base_model.net[4:](h2)
        base_pred = torch.argmax(base_logits, dim=1)

        latent = sae.relu(sae.encoder(h2))
        recon_h2 = sae.decoder(latent)
        sae_logits = base_model.net[4:](recon_h2)
        sae_pred = torch.argmax(sae_logits, dim=1)

    # Metrics
    total = base_pred.numel()
    base_match = torch.sum(base_pred == sae_pred).item() / total * 100
    sae_gt_match = torch.sum(sae_pred == Y_tensor).item() / total * 100
    mse_h2 = torch.nn.functional.mse_loss(recon_h2, h2).item()
    mse_logits = torch.nn.functional.mse_loss(sae_logits, base_logits).item()

    print(f"   ✅ SAE vs BaseModel Prediction Match: {base_match:.2f}%")
    print(f"   ✅ SAE vs Ground Truth Accuracy:      {sae_gt_match:.2f}%")
    print(f"   📏 h2 MSE:                            {mse_h2:.6f}")
    print(f"   📏 Logit MSE:                         {mse_logits:.6f}")

    # Save for global eval
    all_activations.append(h2)
    all_labels.append(Y_tensor)

    # Store metrics
    scenario_metrics.append({
        'scenario': scenario,
        'sae_vs_base': base_match,
        'sae_vs_gt': sae_gt_match,
        'mse_h2': mse_h2,
        'mse_logits': mse_logits
    })

# ------------------ Aggregate Evaluation ------------------
X_all = torch.cat(all_activations, dim=0)
Y_all = torch.cat(all_labels, dim=0)

with torch.no_grad():
    base_logits = base_model.net[4:](X_all)
    base_pred = torch.argmax(base_logits, dim=1)

    latent = sae.relu(sae.encoder(X_all))
    recon_h2 = sae.decoder(latent)
    sae_logits = base_model.net[4:](recon_h2)
    sae_pred = torch.argmax(sae_logits, dim=1)

total = base_pred.numel()
match = torch.sum(base_pred == sae_pred).item()
label_match = torch.sum(Y_all == sae_pred).item()
mse_recon = torch.nn.functional.mse_loss(recon_h2, X_all).item()
mse_logits = torch.nn.functional.mse_loss(sae_logits, base_logits).item()

print("\n🔍 Overall Evaluation on All Training Activations:")
print(f"✅ SAE vs BaseModel Prediction Match: {match}/{total} = {match / total * 100:.2f}%")
print(f"✅ SAE vs Ground Truth Accuracy:      {label_match}/{total} = {label_match / total * 100:.2f}%")
print(f"📏 h2 Reconstruction MSE:             {mse_recon:.6f}")
print(f"📏 Logit Output MSE:                  {mse_logits:.6f}")

# ------------------ Show Mismatches ------------------
diff_indices = torch.where(base_pred != sae_pred)[0]
if len(diff_indices) > 0:
    print("\n⚠️  Example mismatches (first 5):")
    for i in diff_indices[:5]:
        print(f"  Step {i.item():4d}: Base={base_pred[i].item()} | SAE={sae_pred[i].item()} | True={Y_all[i].item()}")
else:
    print("🎉 All predictions match between SAE and base model.")

# ------------------ Plotting ------------------
scenarios_plot = [m['scenario'] for m in scenario_metrics]
acc_base = [m['sae_vs_base'] for m in scenario_metrics]
acc_gt = [m['sae_vs_gt'] for m in scenario_metrics]

x = np.arange(len(scenarios_plot))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, acc_base, width, label='SAE vs Base')
plt.bar(x + width/2, acc_gt, width, label='SAE vs Ground Truth')

plt.ylabel("Accuracy (%)")
plt.title("SAE Prediction Accuracy by Scenario")
plt.xticks(x, scenarios_plot)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
