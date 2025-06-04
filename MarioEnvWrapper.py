import gym
import numpy as np
from collections import deque
from gym.spaces import Box
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
import cv2
import warnings

from configs.config_game import COMPLEX_MOVEMENT


class MarioEnvWrapper(gym.Wrapper):
    def __init__(self,
                 level="SuperMarioBros-v1",
                 movement='simple',
                 grayscale=True,
                 resize_shape=(84, 84),
                 frame_skip=4,
                 frame_stack=4,
                 use_stack=False,  # ✅ 新增：控制是否使用堆叠
                 render_mode=False):
        # 设置动作空间

        self.actions = COMPLEX_MOVEMENT

        env = make(level)
        env = JoypadSpace(env, self.actions)
        super().__init__(env)

        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.use_stack = use_stack  # ✅ 保存标志
        self.render_mode = render_mode

        self.frames = deque(maxlen=frame_stack) if use_stack else None

        channels = 1 if grayscale else 3
        if use_stack:
            obs_shape = (frame_stack, channels, *self.resize_shape)  # (T,C,H,W)
        else:
            obs_shape = (channels, *self.resize_shape)               # (C,H,W)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def preprocess(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = cv2.resize(obs, self.resize_shape)
            obs = np.expand_dims(obs, axis=0)  # (1,H,W)
        else:
            obs = cv2.resize(obs, self.resize_shape)
            obs = np.transpose(obs, (2, 0, 1))  # (3,H,W)
        return obs.astype(np.uint8)

    def _get_stacked_frames(self):
        while len(self.frames) < self.frame_stack:
            self.frames.append(self.frames[0] if len(self.frames) > 0
                               else np.zeros((1 if self.grayscale else 3, *self.resize_shape), dtype=np.uint8))
        return np.stack(self.frames, axis=0)  # (T,C,H,W)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        processed_frame = self.preprocess(obs)

        if self.use_stack:
            self.frames.append(processed_frame)
            obs_out = self._get_stacked_frames()
        else:
            obs_out = processed_frame

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _ = (int(self.env.ram[0x86]) - int(self.env.ram[0x071c])) % 256

        if self.render_mode:
            self.env.render()
        return obs_out, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        processed_frame = self.preprocess(obs)

        if self.use_stack:
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
            obs_out = self._get_stacked_frames()
        else:
            obs_out = processed_frame
        return obs_out
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
