import torch
from torch.utils.data import Dataset
import os
import json
import cv2
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_game import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
You can choose `pure BC` or `BC with penalty` mode as you want.
"""


"""
你希望你的 `MarioDataset` 类能兼容不同层次的 `base_dir` 结构，从而灵活支持：
    * 加载单个实验；
    * 加载一个关卡（包含多个实验）；
    * 加载整个数据集（多个关卡 × 多个实验）；
"""
"""
Assume `base_dir` folder has the following scenario:
    - `DD-HH-MM` folder: contains `frames` folder and trajectory.json.
    - `level` folder: contains `DD-HH-MM` folders.
    - `XXX` folder: custom folder containing `level` folders.
"""

class MarioDataset(Dataset):
    def __init__(self, base_dir, penalty_mode, penalty_N=None, transform=None, frame_stack=4):
        self.samples = []
        self.penalty_mode = penalty_mode
        self.transform = transform
        self.frame_stack = frame_stack
        self.frame_indices = {}  # 用于存储每个实验的帧索引范围

        if penalty_mode and penalty_N is None:
            penalty_N = 15

        def process(traj_path, frame_dir, success: bool):
            with open(traj_path, "r") as f:
                traj = json.load(f)

            # 记录实验的起始和结束索引
            start_idx = len(self.samples)

            if not success and penalty_mode:
                for i in range(min(penalty_N, len(traj))):
                    traj[-1 - i]['is_dead'] = True

            prev_x, prev_y = -1, -1  # 初始位置

            for entry in traj:
                curr_x = entry.get("x_pos", -1)
                curr_y = entry.get("y_pos", -1)
                action = entry["action"]

                # 跳过（未移动）且（待着不动）帧 或者 按了移动但是却没有位置坐标变化
                if curr_x == prev_x and curr_y == prev_y and action == [0, 1, 4]:
                    continue  # 跳过当前帧

                prev_x, prev_y = curr_x, curr_y  # 更新上一次的位置
                img_path = os.path.join(frame_dir, os.path.basename(entry["image_file"]))
                sample = {
                    "image": img_path,
                    "action": entry["action"],
                }

                if penalty_mode:
                    sample["is_critical"] = entry.get("is_dead", False)
                self.samples.append(sample)

            # 记录实验的帧范围
            self.frame_indices[(traj_path, frame_dir)] = (start_idx, len(self.samples))

        # 原始数据加载逻辑保持不变...
        for root, dirs, files in os.walk(base_dir):
            if SUCCESS in files:
                traj_path = os.path.join(root, SUCCESS)
                frame_dir = os.path.join(root, "frames")
                process(traj_path, frame_dir, True)
            elif FAILURE in files:
                traj_path = os.path.join(root, FAILURE)
                frame_dir = os.path.join(root, "frames")
                process(traj_path, frame_dir, False)

        print(f"✅ Loaded {len(self.samples)} samples | Frame stack: {frame_stack}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 找到当前样本所属的实验
        current_exp = None
        for (traj_path, frame_dir), (start, end) in self.frame_indices.items():
            if start <= idx < end:
                current_exp = (traj_path, frame_dir, start, end)
                break

        if current_exp is None:
            raise IndexError("Sample index out of range")

        _, _, start_idx, end_idx = current_exp

        # 计算相对索引
        rel_idx = idx - start_idx

        # 获取帧堆叠范围
        stack_start = max(start_idx, idx - self.frame_stack + 1)
        stack_end = idx + 1
        stack_indices = range(stack_start, stack_end)

        # 加载堆叠帧
        stacked_frames = []
        for i in stack_indices:
            if i < start_idx:  # 如果超出实验范围，用第一帧填充
                entry = self.samples[start_idx]
            else:
                entry = self.samples[i]

            img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing image: {entry['image']}")

            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img)  # 关键修改：转换为PyTorch张量
            if self.transform:
                img_tensor = self.transform(img_tensor)
            stacked_frames.append(img_tensor)

        # 如果帧数不足，用第一帧填充
        while len(stacked_frames) < self.frame_stack:
            stacked_frames.insert(0, stacked_frames[0])


        # 堆叠帧 (C, H, W) -> (frame_stack, H, W)
        stacked_frames = torch.stack(stacked_frames, dim=0)

        # 获取当前帧的动作和状态
        current_entry = self.samples[idx]
        action = torch.tensor(current_entry["action"], dtype=torch.long)

        if self.penalty_mode:
            is_critical = torch.tensor(current_entry['is_critical'], dtype=torch.bool)
            return stacked_frames, action, is_critical
        else:
            return stacked_frames, action


def train_valid_split(dataset, val_ratio=0.2, seed=42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


if __name__ == "__main__":
    base_dir = SUPERVISED_DATA_DIR
    penalty_mode = False  # ✅ 这里显式指定当前模式
    dataset = MarioDataset(base_dir, penalty_mode=penalty_mode)

    train_set, val_set = train_valid_split(dataset, val_ratio=0.2)

    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    # === 根据模式解包样本 ===
    sample = train_set[10]
    if penalty_mode:
        img, action, is_critical = sample
        print(f"Image shape: {img.shape}, Action: {action.item()}, Critical: {is_critical.item()}")
    else:
        img, action = sample
        print(f"Image shape: {img.shape}, Action: {action.item()}")

    # === 可视化图像 ===
    plt.imshow(img[0].squeeze(), cmap="gray")
    plt.title(f"Action: {action.item()}" + (f" | Critical" if penalty_mode and is_critical else ""))
    plt.axis("off")
    plt.show()

