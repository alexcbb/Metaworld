import gymnasium as gym
import metaworld
import matplotlib.pyplot as plt


print(gym.envs.registry.keys())

seed = 42 # for reproducibility

env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed, render_mode="rgb_array") # MT1 with the reach environment

obs, info = env.reset()

for i in range(5):
    a = env.action_space.sample() # randomly sample an action
    obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
    plt.imshow(env.render())
    plt.axis('off')
    plt.show()
