import torch
import time
from MarioEnvWrapper import MarioEnvWrapper  # 你的封装环境类
from configs.config_game import COMPLEX_MOVEMENT, MODEL_TYPE, FRAME_STACK, TEMPORAL_TYPE, USE_STACK
from supervised.model.ModelFactory import ModelFactory

# 可自定义关卡编号
WORLD = 1
STAGE = 1
LEVEL_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-v1"

NUM_ACTIONS = len(COMPLEX_MOVEMENT)


# === 1. 初始化模型 ===
model = ModelFactory.create_model(
    model_type=MODEL_TYPE,
    num_actions=NUM_ACTIONS,
    frame_stack=FRAME_STACK,
    temporal_type=TEMPORAL_TYPE,
)
#state_dict = torch.load("result/11-17-16/weights/best_model.pt", map_location=torch.device('cpu'))
state_dict = torch.load("result/CNN1d/04-16-22/weights/best_model.pt", map_location=torch.device('cpu'), weights_only=True)
#state_dict = torch.load("result/CNN1d/28-12-39/weights/best_model.pt", map_location=torch.device('cpu'), weights_only=True)
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
    frame_stack=FRAME_STACK,
    use_stack=USE_STACK,
)

# === 3. 推理控制循环 ===
done = False
state = env.reset()#[1,84,84] / [8,1,84,84]
print(state.shape)

with torch.no_grad():
    while True:
        # === 格式转换：numpy -> torch ===
        if(USE_STACK):
            state_tensor = torch.from_numpy(state).float() / 255.0
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0
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
