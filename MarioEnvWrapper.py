import gym
import numpy as np
from collections import deque
from gym.spaces import Box
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import cv2
import warnings

class MarioEnvWrapper(gym.Wrapper):
    def __init__(self,
                 level="SuperMarioBros-v0",
                 movement='simple',
                 grayscale=True,
                 resize_shape=(84, 84),
                 frame_skip=4,
                 frame_stack=4,  # 新增帧堆叠参数
                 render_mode=False):
        # 选择动作空间
        if movement == 'simple':
            self.actions = SIMPLE_MOVEMENT
        elif movement == 'complex':
            self.actions = COMPLEX_MOVEMENT
        else:
            self.actions = movement  # 自定义动作列表

        env = make(level)
        env = JoypadSpace(env, self.actions)
        super().__init__(env)

        self.grayscale = grayscale
        self.resize_shape = resize_shape
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack  # 存储帧堆叠数量
        self.render_mode = render_mode

        # 初始化帧堆叠缓冲区
        self.frames = deque(maxlen=frame_stack)

        # 重定义observation space
        channels = 1 if grayscale else 3
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(resize_shape[1], resize_shape[0], channels * frame_stack),  # 修改通道数为堆叠帧数
            dtype=np.uint8
        )

    def preprocess(self, obs):
        """预处理单帧图像"""
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
            obs = np.expand_dims(obs, axis=-1)  # 保持三维结构 (H,W,1)
        else:
            obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)

    def _get_stacked_frames(self):
        """获取堆叠后的帧序列"""
        # 如果缓冲区未满，用第一帧填充
        while len(self.frames) < self.frame_stack:
            self.frames.append(self.frames[0] if len(self.frames) > 0
                               else np.zeros((*self.resize_shape[::-1], 1 if self.grayscale else 3)))
        return np.concatenate(self.frames, axis=-1)  # 沿通道维度堆叠

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        # 执行frame_skip次动作
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        # 预处理当前帧并加入缓冲区
        processed_frame = self.preprocess(obs)
        self.frames.append(processed_frame)

        # 获取堆叠帧
        stacked_obs = self._get_stacked_frames()

        # 修复overflow warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _ = (int(self.env.ram[0x86]) - int(self.env.ram[0x071c])) % 256

        if self.render_mode:
            self.env.render()

        return stacked_obs, total_reward, done, info

    def reset(self):
        """重置环境并初始化帧堆叠"""
        obs = self.env.reset()
        processed_frame = self.preprocess(obs)

        # 清空并重新初始化帧缓冲区
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)

        return self._get_stacked_frames()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()