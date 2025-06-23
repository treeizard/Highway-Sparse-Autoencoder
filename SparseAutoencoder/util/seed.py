# seed.py

import random
import numpy as np

def set_seed(seed: int = 45):
    """
    Set the seed for Python, NumPy, and (optionally) PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Python and NumPy seeds set to {seed}")

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("PyTorch seed set.")
    except ImportError:
        print("PyTorch not installed; skipping torch seed.")

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
