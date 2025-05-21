import gym
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from test_obs import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    env =make_env()
    vec_env = SubprocVecEnv([make_env for _ in range(8)])
    model = PPO("CnnPolicy", env, verbose=1,tensorboard_log='logs')
    model.learn(total_timesteps=1e7)
    model.save("ppo_mario")

if  __name__ =='__main__':
    main()