import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

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

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

pygame.init()
pygame.display.set_mode((1, 1))  # 激活事件系统

clock = pygame.time.Clock()
done = True
running = True

KEY_TO_MARIO_BUTTON = {
    pygame.K_d: 'right',
    pygame.K_a: 'left',
    pygame.K_k: 'A',
    pygame.K_j: 'B',
    pygame.K_w: 'up',
    pygame.K_s: 'down',
}

print("🎮 控制说明：D=右, A=左, K=跳, J=加速, W=上, S=下，支持组合键，如 D+K")

while running:
    if done:
        env.reset()

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
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    _, _, done, _ = env.step(current_action)
    env.render()

    if pressed_buttons:
        print(f"[{time.strftime('%H:%M:%S')}] 当前动作: {COMPLEX_MOVEMENT[current_action]}")
    print(f"当前FPS: {clock.get_fps():.2f}", end="\r")

    clock.tick(60)

env.close()
pygame.quit()
