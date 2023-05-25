import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


class InertiaCarEnv(gym.Env):
    def __init__(self):
        super(InertiaCarEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, -10]), high=np.array([10, 10]), dtype=np.float32)
        
        self.target_radius = 0.5
        self.max_position = 10
        self.min_position = 3
        self.max_speed = 10
        self.min_speed = -10
        self.tau=0.5
        self.dt=0.1
        self.tau_a=np.exp(-self.dt/self.tau)
        
        
    def reset(self):
        self.timeout=66
        self.position = 0
        self.speed = 0
        self.target = np.random.uniform(self.min_position, self.max_position)
        return np.array([self.position, self.speed])

    def step(self, action):
        self.timeout-=1
        action = np.clip(action, -1, 1)[0]
        
        self.speed += action
        self.speed = self.tau_a*self.speed + (1-self.tau_a)*action
       
        self.position += self.speed*self.dt
        self.position = np.clip(self.position, self.min_position, self.max_position)
        
        distance_to_target = abs(self.position - self.target)
        done = distance_to_target <= self.target_radius or self.timeout<0
        reward = 1 if done else -0.1
        
        return np.array([self.position, self.speed]), reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
        self.ax.clear()
        self.ax.set_xlim(self.min_position, self.max_position)
        self.ax.set_ylim(-1, 1)
        self.ax.add_patch(plt.Circle((self.target, 0), self.target_radius, color='r'))
        self.ax.add_patch(plt.Circle((self.position, 0), 0.1, color='b'))
        self.ax.axis('equal')
        plt.show()
        # plt.pause(0.01)


# Example usage
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    total_episodes = 100

    for episode in range(total_episodes):
        obs = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()


    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            obs = env.reset()

            


import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

env = gym.make("CartPole-v1")
env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
for _ in range(100):
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    env.step(action)

