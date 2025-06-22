import gymnasium as gym
import random
import highway_env 
from stable_baselines3 import DQN
from util.seed import set_seed, seed_env

class MultiSceneEnv(gym.Env):
    def __init__(self):
        self.envs = [gym.make("highway-v0"), gym.make("merge-v0"), gym.make("roundabout-v0")]
        self.current_env = None
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, **kwargs):
        self.current_env = random.choice(self.envs)
        return self.current_env.reset(**kwargs)

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode="human"):
        return self.current_env.render(mode=mode)

    def close(self):
        for env in self.envs:
            env.close()

# Set Seed for the Environment and the Numpy
env = MultiSceneEnv()
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
              tensorboard_log="log/super/")

model.learn(int(2e5))
model.save("model/super_dqn")

