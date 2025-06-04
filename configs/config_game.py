# import sys
import os
import random
import warnings

# sys.stdout = open(os.devnull, 'w')  # 彻底屏蔽, 太极端，所有打印都屏蔽
import pygame

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# === 自适应项目根路径，确保路径保存正确 === 消除vscode和pycharm处理相对路径的差异问题
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUPERVISED_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "supervised"))
SUPERVISED_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "supervised", "mario_data"))
RESULT_SAVE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "supervised", "result"))
SUCCESS = "trajectory_success.json"
FAILURE = "trajectory_failure.json"

# model参数
MODEL_TYPE = "CNN1d"    #模型类型，支持["CNN1d", "CNN2d", "TCN"]
TEMPORAL_TYPE = "transformer"  #时序处理器类型，支持["lstm", "conv1d", "transformer"]（仅CNN1d需要）
FRAME_STACK = 4         #输入帧堆叠数量（仅CNN1d/TCN需要）
USE_STACK = True       #是否启用帧堆叠


# 人类视角，游戏窗口大小
TARGET_WIDTH, TARGET_HEIGHT = 720, 640

# env输入的动作空间
COMPLEX_MOVEMENT = [
    ['NOOP'],  # stay
    ['right'],
    ['right', 'A'],
    # ['right', 'B'],
    # ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    # ['left', 'B'],
    # ['left', 'A', 'B'],
    # ['down'],
    # ['up'],
    # ['B']
]

# 监听键盘绑定
KEY_TO_MARIO_BUTTON = {
    pygame.K_d: 'right',
    pygame.K_a: 'left',
    pygame.K_k: 'A',
    pygame.K_j: 'B',
    pygame.K_w: 'up',
    pygame.K_s: 'down',
}

level_dirs = [f for f in os.listdir(SUPERVISED_DATA_DIR)
              if os.path.isdir(os.path.join(SUPERVISED_DATA_DIR, f))]