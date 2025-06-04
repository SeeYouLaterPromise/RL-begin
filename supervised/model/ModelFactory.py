from torch import nn

from supervised.model.CNN1d import CNN1d
from supervised.model.CNN2d import CNN2d
from supervised.model.TCN import TemporalConvNet

"""
    改进后的模型工厂方法，支持动态配置时序处理模块
        
    参数：
        model_type: 模型类型，支持["CNN1d", "CNN2d", "TCN"]
        num_actions: 输出动作空间维度
        frame_stack: 输入帧堆叠数量（仅CNN1d/TCN需要）
        temporal_type: 时序处理器类型，支持["lstm", "conv1d", "transformer"]（仅CNN1d需要）
"""

class ModelFactory:
    @staticmethod
    def create_model(model_type, num_actions, frame_stack=4, temporal_type='lstm'):

        if model_type == "CNN1d":
            # 带有时序处理的1D CNN模型
            model = CNN1d(temporal_type=temporal_type)
            # 添加动作输出头
            model.fc = nn.Sequential(
                nn.Linear(128, num_actions),  # 假设时序模块输出128维
                nn.Softmax(dim=-1)
            )
            return model

        elif model_type == "CNN2d":
            # 标准2D CNN模型（处理单帧）
            return CNN2d(num_actions)

        elif model_type == "TCN":
            # 时序卷积网络配置
            num_inputs = frame_stack
            num_channels = [64, 64, 128]  # 渐进增加通道数
            tcn = TemporalConvNet(
                num_inputs=num_inputs,
                num_channels=num_channels,
                kernel_size=3,
                dropout=0.1
            )
            # 添加动作输出头
            tcn.fc = nn.Sequential(
                nn.Linear(num_channels[-1], num_actions),
                nn.Softmax(dim=-1)
            )
            return tcn

        else:
            raise ValueError(f"未知模型类型: {model_type}。支持: CNN1d/CNN2d/TCN")