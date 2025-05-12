import torch
import time
from MarioBCModel import MarioBCModel  # 你的模型定义
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MarioEnvWrapper import MarioEnvWrapper  # 你的封装环境类
from configs.config_game import SUPERVISED_DIR, COMPLEX_MOVEMENT

# === 1. 初始化模型 ===
num_actions = 13 # len(COMPLEX_MOVEMENT)
model = MarioBCModel(num_actions=num_actions)
model_path = os.path.abspath(os.path.join(SUPERVISED_DIR, "result/10-20-31/weights/best_model.pt"))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # , map_location="cpu"
model.eval()

print("model load over!")

# === 2. 初始化 Mario 环境 ===
env = MarioEnvWrapper(
    movement='complex',
    grayscale=True,
    resize_shape=(84, 84),
    frame_skip=4,
    render_mode=True  # 控制是否实时渲染
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
        # print(action_id)

        # === 环境执行 ===
        state, reward, done, info = env.step(action_id)
        env.render()
        if done:
            break

        # 控制帧率
        time.sleep(1 / 60)

env.close()
