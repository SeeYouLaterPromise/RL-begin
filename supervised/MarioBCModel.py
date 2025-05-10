import torch.nn as nn
import torch.nn.functional as F

class MarioBCModel(nn.Module):
    def __init__(self, num_actions):
        super(MarioBCModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),  # [batch, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # [batch, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # [batch, 64, 7, 7]
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
