import argparse
import gymnasium as gym
import highway_env  # Ensure this is installed
from stable_baselines3 import DQN
from util.seed import set_seed, seed_env

# Define mapping from input name to environment and model path
ENV_CONFIG = {
    "highway": {"env_id": "highway-v0", "model_path": "model/highway_dqn"},
    "merge": {"env_id": "merge-v0", "model_path": "model/merge_dqn"}
}

def main(env_name):
    if env_name not in ENV_CONFIG:
        raise ValueError(f"Unknown environment '{env_name}'. Choose from {list(ENV_CONFIG.keys())}.")

    env_id = ENV_CONFIG[env_name]["env_id"]
    model_path = ENV_CONFIG[env_name]["model_path"]

    env = gym.make(env_id, render_mode="human")
    seed_env(env)
    set_seed()

    model = DQN.load(model_path)

    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DQN model on a highway_env scenario.")
    parser.add_argument("env_name", choices=["highway", "merge"], help="Environment to test ('highway' or 'merge')")
    args = parser.parse_args()
    main(args.env_name)
