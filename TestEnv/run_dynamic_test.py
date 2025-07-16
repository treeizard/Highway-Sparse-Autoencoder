import os
import sys
import gymnasium as gym
import highway_env
import random
import numpy as np
import imageio
import cv2
import torch
from tqdm import trange
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SparseAutoencoder.models.base_model import TorchMLP
from SparseAutoencoder.models.SAE import SparseAutoencoder

# -------------------------------
# Config
# -------------------------------
num_scenarios = 1
record_video = True
play_video = False
max_steps = 5000
model_name = 'super_dqn'
SEED = 45
neuron_index = 16
perturb_factor = 0.7

# -------------------------------
# Set seeds globally
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
set_seed(SEED)

# -------------------------------
# Paths
# -------------------------------
base_model_path = os.path.abspath(os.path.join("..", "BaseModel", "torch_model", f"{model_name}.pt"))
sae_ckpt_path = os.path.abspath(os.path.join("..", "SparseAutoencoder", "sae_ckpt", f"sae_{model_name}_best.pt"))

# -------------------------------
# Load Models
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = TorchMLP().to(device)
base_model.load_state_dict(torch.load(base_model_path, map_location=device))
base_model.eval()

sae = SparseAutoencoder(input_dim=256, hidden_dim=1024).to(device)
sae.load_state_dict(torch.load(sae_ckpt_path, map_location=device))
sae.eval()

# -------------------------------
# Environments to run
# -------------------------------
ENV_LIST = ['highway-v0', 'merge-v0', 'roundabout-v0']

for env_id in ENV_LIST:
    scenario_name = env_id.split('-')[0]
    print(f"\n=== Running {num_scenarios} episodes on {env_id} ===")

    # Output dirs
    # Output dirs
    output_dir = os.path.join("data", model_name + "_sae", scenario_name)
    os.makedirs(os.path.join(output_dir, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "actions"), exist_ok=True)  
    if record_video:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    # Create environment
    env = gym.make(env_id, render_mode='rgb_array' if (record_video or play_video) else None)

    for demo_id in trange(1, num_scenarios + 1, desc=f"{scenario_name}"):
        obs_log, label_log, action_log, frame_log = [], [], [], []

        obs, _ = env.reset(seed=SEED + demo_id)
        done = False
        step = 0

        while not done and step < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)

            with torch.no_grad():
                h1 = base_model.net[:2](obs_tensor)
                h2 = base_model.net[2:4](h1)
                latent = sae.relu(sae.encoder(h2))
                latent[:, neuron_index] += perturb_factor
                recon = sae.decoder(latent)
                q_values = base_model.net[4:](recon)
                action = torch.argmax(q_values, dim=1).item()

            obs_log.append(obs)
            label_log.append(int(action))
            action_log.append(q_values.cpu().numpy())  # <- Save full Q-values per step

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if record_video or play_video:
                frame = env.render()
                if frame is not None:
                    if record_video:
                        frame_log.append(frame)
                    if play_video:
                        cv2.imshow("Simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            done = True
                            break

            step += 1

        # Save demo
        np.save(os.path.join(output_dir, "obs", f"demo{demo_id}.npy"), np.array(obs_log))
        np.save(os.path.join(output_dir, "labels", f"demo{demo_id}.npy"), np.array(label_log))
        np.save(os.path.join(output_dir, "actions", f"demo{demo_id}.npy"), np.array(action_log))  # <- Save actions

        if record_video and len(frame_log) > 0:
            video_path = os.path.join(output_dir, "videos", f"demo{demo_id}.mp4")
            print("Video Path:", video_path)
            imageio.mimsave(video_path, frame_log, fps=16)


    env.close()

if play_video:
    cv2.destroyAllWindows()
