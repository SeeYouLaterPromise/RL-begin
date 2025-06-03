from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_game import *
import cv2
import time
import json
import pygame

# 可自定义关卡编号
WORLD = 1
STAGE = 1
LEVEL_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-v1"

# 获取当前时间并格式化为字符串
experiment_time = time.strftime("%d-%H-%M", time.localtime())
save_dir = f"{SUPERVISED_DATA_DIR}/{WORLD}-{STAGE}/{experiment_time}"
frame_dir = os.path.join(save_dir, "frames")
os.makedirs(frame_dir, exist_ok=True)

# 图像目标尺寸（灰度）k
RESIZE_SHAPE = (84, 84)

# 每隔4帧采集一帧
FRAME_SKIP = 4

def update_save(state, frame_count, trajectory, current_action, is_dead, info):
    # ==== 图像保存：灰度 + 缩放 ==== (无需转置，直接用 state)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
    filename = f"frame_{frame_count:06d}.png"
    filepath = os.path.join(frame_dir, filename)
    cv2.imwrite(filepath, resized)

    # 保存轨迹信息（添加x_pos和y_pos）
    trajectory.append({
        "frame_id": int(frame_count),
        "image_file": filepath,
        "action": int(current_action),
        "timestamp": time.time(),
        "is_dead": bool(is_dead),
        "x_pos": int(info['x_pos']),  # 添加x坐标
        "y_pos": int(info['y_pos'])   # 添加y坐标
    })
    return trajectory

def game_loop():
    # 环境初始化dd
    env = make(LEVEL_NAME)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # pygame初始化
    pygame.init()
    screen = pygame.display.set_mode((TARGET_WIDTH, TARGET_HEIGHT))
    clock = pygame.time.Clock()

    state = env.reset()
    running = True
    total_count = 0
    frame_count = 0
    started_recording = False
    trajectory = []
    status = "NOOP"

    while running:
        if not pygame.key.get_focused():
            continue

        # 构造当前按键组合
        pressed = pygame.key.get_pressed()
        pressed_buttons = set()
        for key, button in KEY_TO_MARIO_BUTTON.items():
            if pressed[key]:
                pressed_buttons.add(button)

        # 查找动作编号
        current_action = 0
        for i, combo in enumerate(COMPLEX_MOVEMENT):
            if set(combo) == pressed_buttons:
                current_action = i
                break

        # 事件退出检测
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("退出游戏")
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # update game frame
        state, reward, done, info = env.step(current_action)
        is_success = info['flag_get']
        status = "success" if is_success else "failure"

        # 结束帧
        if done:
            trajectory = update_save(state, frame_count, trajectory, current_action, is_dead=not is_success, info=info)
            break

        # 渲染显示
        frame = np.transpose(state, (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame)
        scaled = pygame.transform.scale(surface, (TARGET_WIDTH, TARGET_HEIGHT))
        screen.blit(scaled, (0, 0))
        pygame.display.update()
        total_count += 1

        # 检查是否开始采集
        if not started_recording:
            if current_action != 0:
                print('play moving, starting collecting data')
                started_recording = True
            else:
                continue

        # Frame-skipping
        if total_count % FRAME_SKIP == 0:
            trajectory = update_save(state, frame_count, trajectory, current_action, False, info)
            frame_count += 1

        clock.tick(60)

    env.close()
    return trajectory, frame_count, status

if __name__ == "__main__":
    trajectory, frame_count, status = game_loop()

    trajectory_name = f"trajectory_{status}.json"
    json_path = os.path.join(save_dir, trajectory_name)
    with open(json_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"✅ 数据采集完成，共采集 {frame_count} 帧")

    pygame.quit()