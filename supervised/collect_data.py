from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
import numpy as np
import os
from configs.config_game import *
import cv2
import time
import json

# 可自定义关卡编号
WORLD = 1
STAGE = 1
LEVEL_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-v0"

# 获取当前时间并格式化为字符串
experiment_time = time.strftime("%d-%H-%M", time.localtime())

# 数据保存路径
save_dir = f"mario_data/{WORLD}-{STAGE}/{experiment_time}"
trajectory_name = "trajectory.json"
frame_dir = os.path.join(save_dir, "frames")
os.makedirs(frame_dir, exist_ok=True)
trajectory = []
# 图像目标尺寸（灰度）
RESIZE_SHAPE = (84, 84)
# 每隔4帧采集一帧
FRAME_SKIP = 4

# 环境初始化
env = make(LEVEL_NAME)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# pygame初始化
pygame.init()
screen = pygame.display.set_mode((TARGET_WIDTH, TARGET_HEIGHT))
clock = pygame.time.Clock()

done = False
# AssertionError: Cannot call env.step() before calling reset()
state = env.reset()
running = True
total_count = 0
frame_count = 0

while running:
    if not pygame.key.get_focused():
        continue
        # print("⚠️ 窗口失去焦点...")

    if done:
        break

    # 构造当前按键组合 （键盘监听符合定义的键盘按键）
    pressed = pygame.key.get_pressed()
    pressed_buttons = set()
    for key, button in KEY_TO_MARIO_BUTTON.items():
        if pressed[key]:
            pressed_buttons.add(button)

    # 查找动作编号（将键盘按键转换成动作空间的编号）
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

    state, _, done, _ = env.step(current_action)

    # 渲染显示（彩色缩放）- 人类
    frame = np.transpose(state, (1, 0, 2))  # 转为 (宽, 高, 通道)
    surface = pygame.surfarray.make_surface(frame)  # 将 NumPy 图像转为 Pygame 可渲染的对象
    scaled = pygame.transform.scale(surface, (TARGET_WIDTH, TARGET_HEIGHT))  # 放大图像，适配你的 pygame 窗口尺寸
    screen.blit(scaled, (0, 0))  ## 把图像贴到屏幕上（坐标位置（0， 0））
    pygame.display.update()  # 刷新窗口，真正的显示新的一帧

    total_count += 1
    # Frame-skipping
    if total_count % FRAME_SKIP == 0:
        # ==== 图像保存：灰度 + 缩放 ==== (无需转置，直接用 state)
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
        filename = f"frame_{frame_count:06d}.png"
        filepath = os.path.join(frame_dir, filename)
        cv2.imwrite(filepath, resized)

        # 保存轨迹信息
        trajectory.append({
            "frame_id": frame_count,
            "image_file": os.path.join(frame_dir, filename),
            "action": current_action,
            "timestamp": time.time()
        })

        # don't forget it
        frame_count += 1

    clock.tick(60)  # clock.tick(60)

# 保存 JSON 轨迹数据
json_path = os.path.join(save_dir, trajectory_name)
with open(json_path, "w") as f:
    json.dump(trajectory, f, indent=2)

print(f"✅ 数据采集完成，共采集 {total_count} 帧，保存至：{save_dir}")
env.close()
pygame.quit()
