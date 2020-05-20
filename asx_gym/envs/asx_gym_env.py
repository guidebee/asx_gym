import numpy as np
import io
from time import sleep
import cv2
from gym.envs.classic_control import rendering
from matplotlib.image import imread
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding

plt.style.use('seaborn-pastel')


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class AsxGymEnv(gym.Env):

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.step_count = 0

        self.viewer = rendering.SimpleImageViewer()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.ax.clear()
        x = np.linspace(-np.pi, np.pi)
        self.step_count += 1
        self.ax.plot(x, np.sin(x - self.step_count / (2 * np.pi)), label="sin", color='r')
        self.ax.set_title(f"sin(x) with {self.step_count}")
        done = False
        if self.step_count > 360:
            done = True
        return self.step_count, 0, done, {}

    def reset(self):
        x = np.linspace(-np.pi, np.pi)
        self.ax.set_xlim(-np.pi, np.pi)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.ax.plot(x, np.sin(x - self.step_count / (2 * np.pi)), label="sin", color='b')
        self.ax.set_title(f"sin(x) with {self.step_count}")

    def render(self, mode='human'):
        img = get_img_from_fig(self.fig)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            self.viewer.imshow(img)
            return self.viewer.isopen
