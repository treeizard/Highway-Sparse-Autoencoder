import gymnasium as gym
import highway_env  # make sure this is installed
from stable_baselines3 import DQN
from seed import set_seed, seed_env

# Create Environment
env = gym.make("highway-v0")

# Set Seed for the Environment and the Numpy
seed_env(env)
set_seed()

# Set Seed for the Environment and Numpy
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="log/highway_dqn/")

model.learn(int(2e4))
model.save("model/highway_dqn")

