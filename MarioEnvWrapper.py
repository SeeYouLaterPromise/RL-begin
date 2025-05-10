import gym
import numpy as np
from gym.spaces import Box
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import cv2
import warnings

class MarioEnvWrapper(gym.Wrapper):
    def __init__(self,
                 movement='simple',
                 grayscale=True,
                 resize_shape=(84, 84),
                 frame_skip=4,
                 render_mode=False):
        # 选择动作空间
        if movement == 'simple':
            self.actions = SIMPLE_MOVEMENT
        elif movement == 'complex':
            self.actions = COMPLEX_MOVEMENT
        else:
            self.actions = movement  # 自定义动作列表（如 hybrid）

        env = make("SuperMarioBros-v0")
        env = JoypadSpace(env, self.actions)
        super().__init__(env)

        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # observation space 重定义
        channels = 1 if grayscale else 3
        self.observation_space = Box(low=0, high=255,
                                     shape=(resize_shape[1], resize_shape[0], channels),
                                     dtype=np.uint8)

    def preprocess(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
            obs = np.expand_dims(obs, axis=-1)
        else:
            obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        # ⛔ 修复 overflow warning 来源：RAM 运算
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _ = (int(self.env.ram[0x86]) - int(self.env.ram[0x071c])) % 256  # 原始位置

        obs = self.preprocess(obs)
        if self.render_mode:
            self.env.render()
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.preprocess(obs)
        return obs

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
