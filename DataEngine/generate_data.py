import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import numpy as np
import os

import imageio
import cv2
from tqdm import trange

# Settings
num_scenarios = 100
record_video = True
play_video = True
model_path = "model/highway_dqn"
model_name = os.path.splitext(os.path.basename(model_path))[0]

# Load model
model = DQN.load(model_path)

# Env config
gym.register_envs(highway_env)
'''
env_config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "vx", "vy", "cos_h"],
        "normalize": True,
        "absolute": False
    },
    "vehicles_count": 2,
    "initial_spacing": 1.5,
    "duration": 40,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "simulation_frequency": 15,
    "policy_frequency": 5
}
'''
env = gym.make('highway-v0', render_mode='rgb_array' if record_video or play_video else None)
#env.unwrapped.configure(env_config)

# Create output dirs
base_dir = f"data/{model_name}"
os.makedirs(os.path.join(base_dir, "obs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "labels"), exist_ok=True)
if record_video:
    os.makedirs(os.path.join(base_dir, "videos"), exist_ok=True)

# Scenario loop
for demo_id in trange(1, num_scenarios + 1, desc="Generating scenarios"):
    obs_log, label_log, frame_log = [], [], []

    obs, _ = env.reset()
    done = False
    step = 0
    max_steps = 1000

    while not done and step < max_steps:
        obs_log.append(obs)
        frame = env.render() if (record_video or play_video) else None
        if record_video:
            frame_log.append(frame)
        if play_video and frame is not None:
            cv2.imshow("Simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        action, _ = model.predict(obs, deterministic=True)
        label_log.append(int(action))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        step += 1

    np.save(os.path.join(base_dir, "obs", f"demo{demo_id}.npy"), np.array(obs_log))
    np.save(os.path.join(base_dir, "labels", f"demo{demo_id}.npy"), np.array(label_log))
    if record_video:
        imageio.mimsave(os.path.join(base_dir, "videos", f"demo{demo_id}.mp4"), frame_log, fps=15)

if play_video:
    cv2.destroyAllWindows()
env.close()
