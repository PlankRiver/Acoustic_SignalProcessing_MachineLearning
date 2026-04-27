import torch
from torch import nn
import torchaudio
import numpy as np


# 定义 CNN 模型
class AudioCNN(nn.Module):
    def __init__(self, lr=0.01):
        super().__init__()
        # 【修改点1】手动保存超参数，替代 self.save_hyperparameters()
        self.lr = lr

        # 网络结构保持不变
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            # 注意：LazyLinear 需要 PyTorch 版本 >= 1.8
            # 它会在第一次 forward 时自动推断输入维度
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(10)  # 输出 0-9
        )

    # 【修改点2】原生 PyTorch 必须显式定义前向传播路径
    def forward(self, x):
        return self.net(x)