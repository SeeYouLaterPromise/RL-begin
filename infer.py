from stable_baselines3 import PPO
from RLEnvironment import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
import numpy as np


def inference(model_path, total_timesteps=1000, render=True):
    # 创建环境（必须与训练时完全一致）
    env = SubprocVecEnv([make_env])
    env = VecFrameStack(env, 4, channels_order='last')


    try:
        model = PPO.load(model_path, device='cpu')  # 显式指定device
    except RuntimeError:
        # 处理PyTorch 2.0+的安全加载
        import torch
        model = PPO.load(model_path, device='cpu',
                         custom_objects={'torch_load': lambda f, _: torch.load(f, weights_only=True)})

    obs = env.reset()

    for step in range(total_timesteps):
        # 确保观察值形状正确
        if obs.shape != (1, 4, 84, 84):  # 假设训练时是通道优先
            obs = np.transpose(obs, (0, 3, 1, 2))  # 从 (1,84,84,4) 转为 (1,4,84,84)

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        if render:
            env.render(mode='human')

        if dones[0]:
            print(f"Episode finished at step {step}")
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    # 模型路径
    model_paths = [
        "./best_model/best_model.zip"
    ]

    for path in model_paths:
        try:
            print(f"Trying model: {path}")
            inference(model_path=path, total_timesteps=2000)
            break
        except FileNotFoundError:
            print(f"{path} not found, trying next...")