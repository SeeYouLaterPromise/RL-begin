import random
import warnings

import numpy as np
import torch

from MarioEnvWrapper import MarioEnvWrapper
from configs.config_game import MODEL_TYPE, FRAME_STACK, level_dirs, USE_STACK
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ExamManager:
    def __init__(self, model, device='cpu', render=False):
        self.model = model.to(device)

        chosen_levels = random.sample(level_dirs, k=min(len(level_dirs), 1))
        world_stage = chosen_levels[0].split('-')  # ['1', '1']
        WORLD = int(world_stage[0])
        STAGE = int(world_stage[1])
        LEVEL_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-v1"
        if(MODEL_TYPE == "CNN2d"):
            use_stack = False
        else:
            use_stack = True
        env = MarioEnvWrapper(
            level=LEVEL_NAME,
            movement='complex',
            grayscale=True,
            resize_shape=(84, 84),
            frame_skip=4,
            render_mode=False,  # 控制是否实时渲染
            frame_stack=FRAME_STACK,
            use_stack=use_stack,
        )
        self.env = env
        self.device = device
        self.render = render
        self.last_x_pos = 0
        self.stuck_frames = 0
        self.max_stuck_frames = 60  # 假设60帧未移动视为死亡（约2秒，按30FPS计算）

    def run_exam(self, n_episodes=10):
        distances = []
        deaths = 0
        done = False
        successes = 0
        for ep in range(n_episodes):
            has_failed = False
            state = self.env.reset()
            done = False
            x_start = self.env.unwrapped._get_info().get('x_pos', 0)
            x_end = x_start

            while not done:
                if USE_STACK:
                    state_tensor = torch.from_numpy(state).float() / 255.0
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) / 255.0

                with torch.no_grad():
                    logits = self.model(state_tensor.to(self.device))
                    action_id = torch.argmax(logits, dim=1).item()

                next_state, reward, done, info = self.env.step(action_id)

                if self.render:
                    self.env.render()

                x_end = info.get('x_pos', x_end)
                if self.is_failure(info) and not has_failed:
                    has_failed = True

                if not has_failed and info.get("flag_get", False):
                    successes += 1  # 成功条件：未失败且拿到旗帜
                    print("成功通关")
                state = next_state

            distance = x_end - x_start
            distances.append(distance)

        result = {
            "mean_distance": np.mean(distances),
            "pass_rate": successes / n_episodes,
            "distances": distances
        }
        result["exp"] = self.compute_exp(result)
        return result

    def compute_exp(self, result, epoch=0, steps=100):
        # 动态惩罚系数
        death_penalty = 50 * (1 + np.log(epoch+1))
        γ = 0.95  # 折扣因子

        # 多维度计算
        exp = (result["mean_distance"] * γ**steps -
               (1 - result["pass_rate"]) * death_penalty)
        return max(0, exp)

    def adjust_training_plan(self, result):
        exp = result['exp']
        if exp < 100:
            return {"add_bad_case_data": True, "increase_penalty_weight": 0.2}
        elif exp < 300:
            return {"fine_tune": True}
        else:
            return {"save_model": True, "advance_level": True}
    def is_failure(self, info):
        current_x = info.get('x_pos', 0)

        # 位移检测
        if abs(current_x - self.last_x_pos) < 1:  # 位移变化小于1像素
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0
        self.last_x_pos = current_x

        return any([
            info.get("dead", False),
            info.get("time", 400) <= 0,
            info.get("y_pos", 0) < 50,  # 防止掉入悬崖
            self.stuck_frames >= self.max_stuck_frames  # 新增卡死判定
        ])