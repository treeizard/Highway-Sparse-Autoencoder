import gymnasium
import highway_env
from matplotlib import pyplot as plt
import pprint

env = gymnasium.make("highway-v0", render_mode='rgb_array')
pprint.pprint(env.unwrapped.config)

env.unwrapped.config["ego_spacing"] = 4
env.reset()
plt.imshow(env.render())
plt.show()