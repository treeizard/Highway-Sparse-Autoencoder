# seed.py
import torch
import os
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_env(env, seed: int = 45):
    """
    Seed a Gym environment and its action/observation spaces.
    Compatible with both Gym and Gymnasium.
    """
    # New Gym API (Gymnasium and latest Gym versions)
    try:
        env.reset(seed=seed)
    except TypeError:
        # Older Gym API
        env.seed(seed)
    
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, 'seed'):
        env.observation_space.seed(seed)
    
    print(f"Environment seeded with {seed}")
