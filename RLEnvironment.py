
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation
from my_wrapper import SkipFrameWrapper
from gym import Wrapper

class MarioRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 使用env已有的参数，不新建成员变量
        self._reset_state()

    def _reset_state(self):
        # 使用字典存储临时状态，不创建类成员变量
        self.state = {
            'max_x_pos': 0,
            'last_x_pos': 0,
            'stuck_counter': 0,
            'time_counter': 0,
            'jump_penalty': 0,
            'last_action': 0
        }

    def reset(self, **kwargs):
        self._reset_state()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # 使用info中已有的参数
        current_x_pos = info.get('x_pos', 0)
        is_stuck = False

        # 1. 向右移动奖励 (基于现有x_pos参数)
        if current_x_pos > self.state['max_x_pos']:
            x_reward = (current_x_pos - self.state['max_x_pos']) * 0.1
            reward += x_reward
            self.state['max_x_pos'] = current_x_pos
            self.state['stuck_counter'] = 0
        else:
            # 2. 停滞惩罚 (使用现有计数器)
            self.state['stuck_counter'] += 1
            if self.state['stuck_counter'] > 30:
                reward -= 0.1
                is_stuck = True

        # 3. 时间惩罚 (使用现有计数器)
        self.state['time_counter'] += 1
        reward -= 0.01

        # 4. 跳跃相关奖励/惩罚 (使用现有action和last_action)
        if action in [1, 2]:  # 跳跃动作
            if action != self.state['last_action']:
                if abs(current_x_pos - self.state['last_x_pos']) < 1 and is_stuck:
                    self.state['jump_penalty'] = min(1.0, self.state['jump_penalty'] + 0.2)
                    reward -= self.state['jump_penalty']

        # 5. 过关奖励 (使用flag_get参数)
        if info.get('flag_get', False):
            reward += 15  # 使用原文件中的奖励幅度

        # 6. 死亡惩罚 (使用done和flag_get参数)
        if done and not info.get('flag_get', False):
            reward -= 10  # 使用原文件中的惩罚幅度

        # 更新状态 (使用现有参数)
        self.state['last_x_pos'] = current_x_pos
        self.state['last_action'] = action

        return obs, reward, done, info


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrameWrapper(env, skip=4)
    env = GrayScaleObservation(env,keep_dim=True)
    env = ResizeObservation(env, shape=(84,84))
    return env

if __name__ == '__main__':
    env = make_env()

    done = True
    for step in range(4):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        plt.imshow(state,cmap='gray')
        plt.show()

    env.close()
