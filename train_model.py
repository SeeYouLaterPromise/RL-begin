
from stable_baselines3 import PPO
from RLEnvironment import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack
from stable_baselines3.common.callbacks import  EvalCallback



def main():
    vec_env = SubprocVecEnv([make_env for _ in range(8)])
    vec_env = VecFrameStack(vec_env, 4,channels_order='last')
    eva_callback = EvalCallback(vec_env, best_model_save_path="./best_model/",
                                log_path="./callback_logs/",eval_freq=10000//8)

    model_params = {
        'learning_rate': 1e-4, # 学习率
        'ent_coef':0.15,#熵项系数，影响探索性
        'clip_range':0.15,  # 截断范围
        "target_kl": 0.15,  # 设置KL散度早停阙值
        'n_epochs': 10,  # 更新次数

        'n_steps': 2048,  # 每个环境每次更新的步数
        'batch_size': 2048,  # 随机抽取多少数据
        'gamma': 0.97,  # 短视或者长远

        # Log
        'tensorboard_log': r'logs',
        'verbose': 1,
        'policy': "CnnPolicy"
    }
    # model = PPO(env= vec_env,**model_params)

    model=PPO.load('best_model/第二轮best model/best_model.zip', env=vec_env, **model_params)
    model.learn(total_timesteps=int(1e7),callback=eva_callback)

if  __name__ =='__main__':
    main()