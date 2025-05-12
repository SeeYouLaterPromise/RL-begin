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
    def __init__(self, base_dir, penalty_mode, penalty_N=None, transform=None):
        self.samples = []
        self.penalty_mode = penalty_mode
        self.transform = transform

        # robust logic
        if penalty_mode and penalty_N is None:
            penalty_N = 15

        def process(traj_path, frame_dir, success: bool):
            # 读取该实验的 trajectory.json
            with open(traj_path, "r") as f:
                traj = json.load(f)

            # 对于未通关数据集：标记倒数N帧为关键帧 (penalty model is True)
            if not success and penalty_mode:
                for i in range(min(penalty_N, len(traj))):
                    traj[-1 - i]['is_dead'] = True


            # 记录所有样本（全路径）
            for entry in traj:
                img_path = os.path.join(frame_dir, os.path.basename(entry["image_file"]))
                sample = {
                    "image": img_path,
                    "action": entry["action"]
                }

                # penalty mode or not
                if penalty_mode:
                    sample["is_critical"] = entry.get("is_dead", False)
                
                # append
                self.samples.append(sample)

        # 遍历 base_dir，查找所有 trajectory.json
        for root, dirs, files in os.walk(base_dir):
            if SUCCESS in files:
                traj_path = os.path.join(root, SUCCESS)
                frame_dir = os.path.join(root, "frames")
                process(traj_path, frame_dir, True)
                
            elif FAILURE in files:
                traj_path = os.path.join(root, FAILURE)
                frame_dir = os.path.join(root, "frames")
                process(traj_path, frame_dir, False)

        # debug info print        
        print(f"✅ Loaded {len(self.samples)} samples from {base_dir} | penalty_mode={penalty_mode}, penalty_N={penalty_N}")

    

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing image: {entry['image']}")

        img = img.astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        action = torch.tensor(entry["action"], dtype=torch.long)
        if self.penalty_mode:
            is_critical = torch.tensor(entry['is_critical'], dtype=torch.bool)
            return img, action, is_critical
        else:
            return img, action


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
    plt.imshow(img.squeeze(0), cmap="gray")
    plt.title(f"Action: {action.item()}" + (f" | Critical" if penalty_mode and is_critical else ""))
    plt.axis("off")
    plt.show()

