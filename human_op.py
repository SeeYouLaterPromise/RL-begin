import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
from configs.config_game import *


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

pygame.init()
screen = pygame.display.set_mode((TARGET_WIDTH, TARGET_HEIGHT))
clock = pygame.time.Clock()

done = True
running = True

print("🎮 控制说明：D=右, A=左, K=跳, J=加速, W=上, S=下，支持组合键，如 D+K")

while running:
    if not pygame.key.get_focused():
        continue
        # print("⚠️ 窗口失去焦点...")

    if done:
        state = env.reset()

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

    state, _, done, _ = env.step(current_action)

    frame = np.transpose(state, (1, 0, 2))  # 转为 (宽, 高, 通道)
    surface = pygame.surfarray.make_surface(frame)  # 将 NumPy 图像转为 Pygame 可渲染的对象
    scaled = pygame.transform.scale(surface, (TARGET_WIDTH, TARGET_HEIGHT))  # 放大图像，适配你的 pygame 窗口尺寸
    screen.blit(scaled, (0, 0))  ## 把图像贴到屏幕上（坐标位置（0， 0））
    pygame.display.update()  # 刷新窗口，真正的显示新的一帧

    # if pressed_buttons:
    #     print(f"[{time.strftime('%H:%M:%S')}] 当前动作: {COMPLEX_MOVEMENT[current_action]}")
    # print(f"当前FPS: {clock.get_fps():.2f}", end="\r")

    clock.tick(60)

env.close()
pygame.quit()
