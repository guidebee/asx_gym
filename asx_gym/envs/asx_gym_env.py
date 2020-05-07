import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class AsxGymEnv(gym.Env):

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass

    def reset(self):
        pass
