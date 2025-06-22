import argparse
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import os
import json
import numpy as np
from datetime import datetime
from util.seed import set_seed  # Only once globally

# Define environment config
ENV_CONFIG = {
    "highway": {"env_id": "highway-v0"},
    "merge": {"env_id": "merge-v0"},
    "roundabout": {"env_id": "roundabout-v0"}
}

MODEL_PATHS = {
    "highway": "model/highway_dqn",
    "merge": "model/merge_dqn",
    "roundabout": "model/roundabout_dqn",
    "super": "model/super_dqn"
}

def run_one_episode(env_name, model_path, trial_index):
    env_id = ENV_CONFIG[env_name]["env_id"]
    video_folder = f"output/trial_{trial_index}/videos"
    os.makedirs(video_folder, exist_ok=True)

    # Use Monitor wrapper to record video
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_folder,
        name_prefix=f"{env_name}_trial{trial_index}",
        episode_trigger=lambda x: True  # Always record
    )

    # No fixed seed here: allow stochasticity
    model = DQN.load(model_path)

    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "env_name": env_name,
        "trial": trial_index
    }

    done = truncated = False
    obs, info = env.reset(seed=None)  # Use random seed
    episode_data["observations"].append(obs.tolist())

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        episode_data["actions"].append(int(action))
        episode_data["rewards"].append(float(reward))
        episode_data["dones"].append(done or truncated)
        episode_data["observations"].append(obs.tolist())

    env.close()

    # Save JSON
    out_dir = f"output/trial_{trial_index}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{env_name}.json"), "w") as f:
        json.dump(episode_data, f, indent=2)

def main(env_name, model_type):
    # Only set seed once for model reproducibility, not trial determinism
    set_seed()

    trials = 2
    if env_name == "all":
        for i in range(trials):
            for name in ENV_CONFIG:
                print(f"\n--- Trial {i} | Testing {model_type} model on {name} ---")
                model_path = MODEL_PATHS[model_type if model_type == "super" else name]
                run_one_episode(name, model_path, i)
    else:
        if env_name not in ENV_CONFIG:
            raise ValueError(f"Unknown environment '{env_name}'. Choose from {list(ENV_CONFIG.keys()) + ['all']}")
        for i in range(trials):
            print(f"\n--- Trial {i} | Testing {model_type} model on {env_name} ---")
            model_path = MODEL_PATHS[model_type if model_type == "super" else env_name]
            run_one_episode(env_name, model_path, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DQN model on a highway_env scenario.")
    parser.add_argument("env_name", choices=list(ENV_CONFIG.keys()) + ["all"], help="Environment to test")
    parser.add_argument("--model", choices=["super", "default"], default="default",
                        help="Model type to use: 'super' (shared model) or 'default' (per-environment)")
    args = parser.parse_args()
    main(args.env_name, args.model)
