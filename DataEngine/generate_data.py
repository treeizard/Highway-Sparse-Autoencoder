import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import numpy as np
import os
import imageio
import cv2
from tqdm import trange

# Config
num_scenarios = 100
record_video = True
play_video = False  # Set True to display live
max_steps = 5000
model_name = 'super_dqn'

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseModel', 'model'))
model_path = os.path.join(base_dir, model_name)

# Environments to run
ENV_LIST = ['highway-v0', 'merge-v0', 'roundabout-v0']

# Load super model
model = DQN.load(model_path)

for env_id in ENV_LIST:
    scenario_name = env_id.split('-')[0]
    print(f"\n=== Running {num_scenarios} episodes on {env_id} ===")

    # Create output dirs
    output_dir = os.path.join("data", model_name, scenario_name)
    os.makedirs(os.path.join(output_dir, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    if record_video:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    # Create environment
    env = gym.make(env_id, render_mode='rgb_array' if (record_video or play_video) else None)

    for demo_id in trange(1, num_scenarios + 1, desc=f"{scenario_name}"):
        obs_log, label_log, frame_log = [], [], []
        obs, _ = env.reset(seed=None)  # ensure randomness
        done = False
        step = 0

        while not done and step < max_steps:
            obs_log.append(obs)
            if record_video or play_video:
                frame = env.render()
                if record_video:
                    frame_log.append(frame)
                if play_video:
                    cv2.imshow("Simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        done = True
                        break

            action, _ = model.predict(obs, deterministic=True)
            label_log.append(int(action))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        # Save logs
        np.save(os.path.join(output_dir, "obs", f"demo{demo_id}.npy"), np.array(obs_log))
        np.save(os.path.join(output_dir, "labels", f"demo{demo_id}.npy"), np.array(label_log))
        if record_video:
            video_path = os.path.join(output_dir, "videos", f"demo{demo_id}.mp4")
            imageio.mimsave(video_path, frame_log, fps=15)

    env.close()

if play_video:
    cv2.destroyAllWindows()
