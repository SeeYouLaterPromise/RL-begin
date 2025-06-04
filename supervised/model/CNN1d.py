import torch
import torch.nn as nn
import math

"""
视频时序处理模型完整实现
核心功能：处理(B,T,1,84,84)格式的视频帧序列，输出时序特征
包含三大模块：
1. CNN空间特征提取（处理单帧图像）
2. 时序聚合模块（三选一：LSTM/Conv1D/Transformer）
3. 位置编码模块（仅Transformer使用）
"""

# 基础CNN编码器（空间特征提取）
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # 输入: (1, 84, 84)
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # 输出: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 输出: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 输出: (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, T, D)
        B, T = x.shape[:2]
        return self.net(x.reshape(B*T, *x.shape[2:])).reshape(B, T, -1)
#时序聚合模块（三选一）
#LSTM时序聚合
class LSTMTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1])  # 取最后时间步

#1D卷积时序聚合
class Conv1DTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        return self.net(x.permute(0,2,1)).squeeze(-1)

#Transformer时序聚合
class TransformerTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(1,0,2)  # (T, B, D)
        x = self.pos_encoder(x)
        out = self.transformer(x)[-1]  # 取最后时间步
        return self.proj(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (T, B, D)
        # self.pe: (max_len, D)
        return x + self.pe[:x.size(0), :].unsqueeze(1).to(x.device)


class CNN1d(nn.Module):
    def __init__(self, temporal_type='lstm'):
        super().__init__()
        self.encoder = CNNEncoder()

        temporal_modules = {
            'lstm': LSTMTemporal,
            'conv1d': Conv1DTemporal,
            'transformer': TransformerTemporal
        }
        self.temporal = temporal_modules[temporal_type](256)

    def forward(self, x):
        # 输入: (B, T, C, H, W) 或 (T, C, H, W)
        if x.dim() == 4:  # 单样本 (T,C,H,W)
            x = x.unsqueeze(0)  # -> (1,T,C,H,W)

        features = self.encoder(x)  # (B,T,D)
        return self.temporal(features)  # (B,D')