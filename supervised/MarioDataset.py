import torch
from torch.utils.data import Dataset, Subset
import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import sys

# 添加配置文件路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_game import *

# 避免 OpenMP 多线程冲突的问题（针对部分机器）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
        :param base_dir: 数据目录路径（可为单个实验、关卡目录或整个数据集）
        :param penalty_mode: 是否启用惩罚机制（对失败轨迹的末N帧做标记）
        :param penalty_N: 惩罚帧数量（默认为15）
        :param transform: 图像增强转换
        :param use_stack: 是否启用帧堆叠模式
        :param frame_stack: 每个样本堆叠的帧数
        :param skip_static: 是否跳过“未移动”的静态帧
        """

class MarioDataset(Dataset):
    def __init__(self, base_dir, penalty_mode=False, penalty_N=None, transform=None,
                 use_stack=False, frame_stack=4, skip_static=True):

        self.samples = []
        self.penalty_mode = penalty_mode
        self.transform = transform
        self.use_stack = use_stack
        self.frame_stack = frame_stack
        self.skip_static = skip_static
        self.frame_indices = {}  # 存储每个实验的帧起止索引，用于堆叠处理

        if penalty_mode and penalty_N is None:
            penalty_N = 15  # 设置默认惩罚帧数量

        def process(traj_path, frame_dir, success: bool):
            """
            加载单个实验轨迹并解析
            """
            with open(traj_path, "r") as f:
                traj = json.load(f)

            start_idx = len(self.samples)  # 当前实验在samples中的起始索引

            # 如果是失败轨迹且启用惩罚机制，标记末尾N帧为关键帧
            if not success and penalty_mode:
                for i in range(min(penalty_N, len(traj))):
                    traj[-1 - i]['is_dead'] = True

            prev_x, prev_y = -1, -1  # 初始坐标
            for entry in traj:
                curr_x = entry.get("x_pos", -1)
                curr_y = entry.get("y_pos", -1)
                action = entry["action"]

                # 可选：跳过静态帧（未移动 且 动作为站立/平移）
                if self.skip_static and curr_x == prev_x and curr_y == prev_y and action in [1, 4]:
                    continue

                prev_x, prev_y = curr_x, curr_y

                # 构建样本项
                img_path = os.path.join(frame_dir, os.path.basename(entry["image_file"]))
                sample = {
                    "image": img_path,
                    "action": entry["action"],
                    "x_pos": curr_x,
                    "y_pos": curr_y
                }
                if penalty_mode:
                    sample["is_critical"] = entry.get("is_dead", False)
                self.samples.append(sample)

            # 记录该实验的帧范围
            self.frame_indices[(traj_path, frame_dir)] = (start_idx, len(self.samples))

        # 遍历目录，加载所有包含 SUCCESS/FAILURE 的实验
        for root, dirs, files in os.walk(base_dir):
            if SUCCESS in files:
                process(os.path.join(root, SUCCESS), os.path.join(root, "frames"), success=True)
            elif FAILURE in files:
                process(os.path.join(root, FAILURE), os.path.join(root, "frames"), success=False)

        print(f"✅ Loaded {len(self.samples)} samples | Stack: {self.use_stack} | Frame stack: {self.frame_stack}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回样本，支持：
        - 普通模式（返回1帧图像）
        - 堆叠模式（返回frame_stack帧堆叠图像）
        """
        if not self.use_stack:
            # 普通模式：加载单帧图像
            entry = self.samples[idx]
            img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing image: {entry['image']}")
            img = torch.tensor(img.astype("float32") / 255.0).unsqueeze(0)  # (1, H, W)
            if self.transform:
                img = self.transform(img)

            action = torch.tensor(entry["action"], dtype=torch.long)
            if self.penalty_mode:
                is_critical = torch.tensor(entry['is_critical'], dtype=torch.bool)
                return img, action, is_critical
            else:
                return img, action

        # 堆叠模式
        current_exp = None
        for (traj_path, frame_dir), (start, end) in self.frame_indices.items():
            if start <= idx < end:
                current_exp = (start, end)
                break

        if current_exp is None:
            raise IndexError("Sample index out of range")

        start_idx, end_idx = current_exp
        stack_start = max(start_idx, idx - self.frame_stack + 1)
        stack_indices = range(stack_start, idx + 1)

        # 加载帧堆叠
        stacked_frames = []
        for i in stack_indices:
            entry = self.samples[start_idx] if i < start_idx else self.samples[i]
            img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Missing image: {entry['image']}")
            img_tensor = torch.tensor(img.astype("float32") / 255.0)
            if self.transform:
                img_tensor = self.transform(img_tensor)
            stacked_frames.append(img_tensor)

        # 若帧不足，则重复首帧填充
        while len(stacked_frames) < self.frame_stack:
            stacked_frames.insert(0, stacked_frames[0])

        stacked_frames = torch.stack(stacked_frames, dim=0).unsqueeze(1)  # (N, 1, H, W)

        current_entry = self.samples[idx]
        action = torch.tensor(current_entry["action"], dtype=torch.long)
        if self.penalty_mode:
            is_critical = torch.tensor(current_entry['is_critical'], dtype=torch.bool)
            return stacked_frames, action, is_critical
        else:
            return stacked_frames, action


def train_valid_split(dataset, val_ratio=0.2, seed=42):
    """
    将数据集按比例划分为训练集和验证集
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


if __name__ == "__main__":
    base_dir = SUPERVISED_DATA_DIR
    penalty_mode = False  # 是否使用惩罚关键帧标记
    use_stack = True      # 是否启用帧堆叠模式
    dataset = MarioDataset(base_dir, penalty_mode=penalty_mode, use_stack=use_stack)

    train_set, val_set = train_valid_split(dataset)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    # 示例样本解包与可视化
    sample = train_set[10]
    if penalty_mode:
        img, action, is_critical = sample
        print(f"Image shape: {img.shape}, Action: {action.item()}, Critical: {is_critical.item()}")
    else:
        img, action = sample
        print(f"Image shape: {img.shape}, Action: {action.item()}")

    # 可视化第0通道图像
    plt.imshow(img[0].squeeze(), cmap="gray")
    plt.title(f"Action: {action.item()}")
    plt.axis("off")
    plt.show()
