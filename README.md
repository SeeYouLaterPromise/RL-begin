一、项目概述
本项目基于stablebaseline3强化学习框架和gym游戏环境库，实现强化学习算法在经典游戏《超级马里奥》中的应用。
通过训练智能体，使其能够在马里奥游戏环境中自主学习并做出决策，完成关卡挑战。

二、技术架构
2.1 核心框架
stablebaseline3：强大的强化学习库，提供了多种成熟的强化学习算法实现，如 PPO、DQN 等，便于快速搭建和训练智能体。
gym：开源的游戏环境库，本项目使用其马里奥游戏环境，为智能体提供训练和测试的场景。
2.2 算法选择
项目采用PPO作为核心算法，该算法在优化策略网络的同时，有效控制策略更新的幅度，能够在复杂游戏环境中稳定学习，实现高效的智能体训练。



1. 创建新环境：
 conda create -n RL_mario_for_learn python=3.12.3

2. 激活新环境：
 conda activate RL_mario_for_learn

 查看所有已安装包：

 conda list

3. 安装所需要的包：
 为了支持低版本的gym和sb3，对setuptools、wheel的版本进行降低：

 pip install setuptools==65.5.0

 pip install wheel==0.38.4

 降低pip的版本

 python.exe -m pip install pip==20.2.4

 接下来可以安装所有依赖了：

 进入带有requirements.txt文件的文件夹中：

 cd D:\D:\PycharmProject\RL-begin

 安装：

 pip install -r requirements.txt


4. 安装tensorboard
 TensorBoard 是一个由 TensorFlow 提供的可视化工具，主要用于帮助开发者监控、分析和调试机器学习模型的训练过程。它通过直观的图表和交互式界面，展示模型训练中的各种指标和数据，帮助开发者更好地理解模型的行为和性能。

 重点是实时观测。

 pip install tensorboard
 

5. 跳帧操作实现
通过SkipFrameWrapper类实现游戏中的跳帧操作，减少不必要的计算，提升训练效率。
class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip  # 每次step调用时重复执行动作的次数

6. 环境创建
使用多个包装器对原始游戏环境进行处理，包括动作空间简化、跳帧、灰度转换和尺寸调整。
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrameWrapper(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84, 84))
    return env

7. 训练配置
使用stablebaseline3中的PPO算法进行智能体训练，
通过SubprocVecEnv和VecFrameStack实现多环境并行训练与帧堆叠，同时设置评估回调函数来保存最佳模型。

8. 训练过程监控
在训练过程中，可通过tensorboard命令查看训练日志和模型性能指标：
tensorboard --logdir=logs

9. 项目目录结构

.
├── README.md         # 项目说明文档
├── my_wrapper.py     # 自定义跳帧包装器代码
├── RLEnvironment.py  # 环境创建代码
├── train.py          # 智能体训练代码
├── best_model        # 保存最佳模型的目录
├── callback_logs     # 评估回调日志目录
├── logs              # 训练日志目录
├── requirements.txt  # 项目依赖清单


