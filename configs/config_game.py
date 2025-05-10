# import sys
# import os
# sys.stdout = open(os.devnull, 'w')  # 彻底屏蔽, 太极端，所有打印都屏蔽
import pygame

# supervised folder 下面数据集文件夹名称
SUPERVISED_DATA_DIR = "mario_data"

# 人类视角，游戏窗口大小
TARGET_WIDTH, TARGET_HEIGHT = 720, 640

# env输入的动作空间
COMPLEX_MOVEMENT = [
    ['NOOP'],  # stay
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
    ['B']
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