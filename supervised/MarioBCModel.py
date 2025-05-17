import torch.nn as nn
import torch.nn.functional as F

class MarioBCModel(nn.Module):
    def __init__(self, num_actions, frame_stack=4):  # 添加frame_stack参数
        super(MarioBCModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, 8, stride=4),  # 输入通道改为frame_stack
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
