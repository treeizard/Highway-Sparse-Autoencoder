import argparse
import torch
import torch.nn as nn
from stable_baselines3 import DQN
import os

# Custom MLP (adjust input/output sizes if needed)
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

# Convert and save function
def convert_and_save(model_name):
    sb3_path = f"model/{model_name}_dqn"
    torch_path = f"torch_model/{model_name}_dqn.pt"

    model = DQN.load(sb3_path)
    torch_model = TorchMLP()

    with torch.no_grad():
        sb3_layers = [l for l in model.q_net.q_net if isinstance(l, nn.Linear)]
        torch_layers = [l for l in torch_model.net if isinstance(l, nn.Linear)]

        for torch_layer, sb3_layer in zip(torch_layers, sb3_layers):
            torch_layer.weight.copy_(sb3_layer.weight)
            torch_layer.bias.copy_(sb3_layer.bias)

    os.makedirs("torch_model", exist_ok=True)
    torch.save(torch_model.state_dict(), torch_path)
    print(f"✅ Saved Torch model to: {torch_path}")

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SB3 DQN model to Torch")
    parser.add_argument("model_name", choices=["highway", "merge"], help="Model prefix name (e.g., 'highway', 'merge')")
    args = parser.parse_args()

    convert_and_save(args.model_name)
