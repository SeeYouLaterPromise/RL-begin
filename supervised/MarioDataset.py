import torch
from torch.utils.data import Dataset
import os
import json
import cv2
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt

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
    def __init__(self, base_dir, transform=None):
        self.samples = []
        self.transform = transform

        # 遍历 base_dir，查找所有 trajectory.json
        for root, dirs, files in os.walk(base_dir):
            if "trajectory.json" in files:
                traj_path = os.path.join(root, "trajectory.json")
                frame_dir = os.path.join(root, "frames")

                # 读取该实验的 trajectory.json
                with open(traj_path, "r") as f:
                    traj = json.load(f)

                # 记录所有样本（全路径）
                for entry in traj:
                    img_path = os.path.join(frame_dir, os.path.basename(entry["image_file"]))
                    self.samples.append({
                        "image": img_path,
                        "action": entry["action"]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        img = img.astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        action = torch.tensor(entry["action"], dtype=torch.long)
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
    base_dir = "mario_data"  # or "supervised/mario_data/1-1/10-12-37"
    dataset = MarioDataset(base_dir)

    train_set, val_set = train_valid_split(dataset, val_ratio=0.2)

    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    img, action = train_set[0]
    print(f"Image shape: {img.shape}, Action: {action.item()}")

    # 可视化灰度图像
    plt.imshow(img.squeeze(0), cmap="gray")  # squeeze 去掉 channel=1，适配 imshow
    plt.title(f"Action: {action.item()}")
    plt.axis("off")
    plt.show()
