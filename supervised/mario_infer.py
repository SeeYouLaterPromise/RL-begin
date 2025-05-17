import torch
import time
from MarioBCModel import MarioBCModel  # 你的模型定义
from MarioEnvWrapper import MarioEnvWrapper  # 你的封装环境类
from configs.config_game import COMPLEX_MOVEMENT

# 可自定义关卡编号
WORLD = 1
STAGE = 1
LEVEL_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-v0"

FRAME_STACK = 8  #帧堆叠数

# === 1. 初始化模型 ===
num_actions = len(COMPLEX_MOVEMENT)
model = MarioBCModel(num_actions=num_actions, frame_stack=FRAME_STACK)
#state_dict = torch.load("result/11-17-16/weights/best_model.pt", map_location=torch.device('cpu'))
state_dict = torch.load("result/17-19-24/weights/last_model.pt", map_location=torch.device('cpu'),weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print("model load over!")

# === 2. 初始化 Mario 环境 ===

env = MarioEnvWrapper(
    level=LEVEL_NAME,
    movement='complex',
    grayscale=True,
    resize_shape=(84, 84),
    frame_skip=4,
    render_mode=True,  # 控制是否实时渲染
    frame_stack=FRAME_STACK
)

# === 3. 推理控制循环 ===
done = False
state = env.reset()  # (84, 84, 1)
# print(state.shape)

with torch.no_grad():
    while True:
        # === 格式转换：numpy -> torch ===
        state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        # shape: (1, 1, 84, 84)

        # === 模型推理动作 ===
        logits = model(state_tensor)
        action_id = torch.argmax(logits, dim=1).item()
        print(action_id)

        # === 环境执行 ===
        state, reward, done, info = env.step(action_id)
        env.render()
        if done:
            break

        # 控制帧率
        time.sleep(1 / 60)

env.close()
