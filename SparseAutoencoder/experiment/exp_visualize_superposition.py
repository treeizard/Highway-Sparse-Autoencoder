import os
import torch
import matplotlib.pyplot as plt
from models.base_model import TorchMLP
from util.data_tool import load_npy_pairs
import seaborn as sns

# Define Config
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

# === Superposition Analysis: Input Feature Interference ===
# Step 1: Get first layer weights
W = base_model.net[0].weight.detach().clone()  # [256, 25]
W = W.T  # [25, 256] → each row is a feature's vector

# Step 2: Normalize each feature vector
W_norm = W / (1e-5 + torch.norm(W, dim=1, keepdim=True))  # [25, 256]

# Step 3: Compute interference between features
interference = W_norm @ W_norm.T  # [25, 25]
interference.fill_diagonal_(0)

# Step 4: Compute polysemanticity for each input feature
polysemanticity = torch.linalg.norm(interference, dim=-1) # [25]
interference.fill_diagonal_(1)

# Step 5: Visualize polysemanticity
plt.figure(figsize=(10, 4))
plt.bar(range(len(polysemanticity)), polysemanticity.cpu())
plt.xlabel("Input Feature Index")
plt.ylabel("Polysemanticity (L2 norm of interference)")
plt.title("Input Feature Polysemanticity in First MLP Layer")
plt.tight_layout()
plt.show()

# Step 6: Visualize interference matrix
plt.figure(figsize=(6, 5))
sns.heatmap(interference.cpu().numpy(), cmap="coolwarm", center=0)
plt.title("Feature Interference (Dot Product of Normalized Input Vectors)")
plt.xlabel("Feature g")
plt.ylabel("Feature f")
plt.tight_layout()
plt.show()
