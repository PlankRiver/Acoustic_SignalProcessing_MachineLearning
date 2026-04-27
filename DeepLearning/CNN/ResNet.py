import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time

# 1. 设置设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 2. 数据准备 (保持不变，ResNet 也推荐用 224x224 输入)
batch_size = 128
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# -----------------------------------------------------------------
# 3. 定义 ResNet 模型组件
# -----------------------------------------------------------------

class Residual(nn.Module):
    """残差块"""

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 第一个卷积层，可能进行下采样 (stride > 1)
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)

        # 1x1 卷积用于调整 shortcut 的通道数或分辨率
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """构建一个 ResNet Stage，包含 num_residuals 个残差块"""
    blk = []
    for i in range(num_residuals):
        # 如果是该 Stage 的第一个块，且不是整个网络的第一个 Stage，则需要下采样 (stride=2)
        # 且需要用 1x1 卷积调整 shortcut 的维度
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 其他情况（包括第一个 Stage 的第一个块）保持 stride=1
            # 注意：如果是 Stage 的非第一个块，输入通道已经是 num_channels 了
            in_ch = input_channels if (i == 0) else num_channels
            # 特殊情况：如果是第一个 Stage (64->64)，虽然 stride=1，但不需要 1x1 conv
            # 除非输入输出通道不一致 (比如你想修改架构)
            use_1x1 = (in_ch != num_channels)
            blk.append(Residual(in_ch, num_channels, use_1x1conv=use_1x1, strides=1))

    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        # --- Stage 1: 类似于 VGG 的预处理部分 (Stem) ---
        # 输入 1 通道 (FashionMNIST)，输出 64 通道
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # --- Stage 2-5: 残差模块 ---
        self.stages = nn.Sequential()
        # 这里的 arch 格式类似: ((2, 64), (2, 128), (2, 256), (2, 512))
        # 分别代表：(块的数量, 输出通道数)

        in_channels = 64  # b1 出来的通道数
        for i, (num_residuals, num_channels) in enumerate(arch):
            # 只有第一个 Stage (i==0) 不需要下采样
            first_block = (i == 0)
            self.stages.add_module(f"stage_{i + 1}",
                                   resnet_block(in_channels, num_channels, num_residuals, first_block)
                                   )
            in_channels = num_channels  # 更新下一层的输入通道数

        # --- Output Layer ---
        # ResNet 使用全局平均池化，而不是 VGG 的 Flatten + 大全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, 10)  # 这里的 in_channels 是最后一个 Stage 的输出 (512)

    def forward(self, x):
        x = self.b1(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# 4. 实例化 ResNet-18
# ResNet-18 架构: 4 个 Stage，每个 Stage 有 2 个残差块
# 通道数序列: 64 -> 128 -> 256 -> 512
resnet18_arch = ((2, 64), (2, 128), (2, 256), (2, 512))
model = ResNet(resnet18_arch).to(device)


# -----------------------------------------------------------------

# 5. 初始化权重
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model.apply(init_weights)

# 查看每一层的形状（验证网络结构）
print("\nChecking Model Layout:")
X_dummy = torch.randn(1, 1, 224, 224).to(device)
# 手动通过每一部分来打印形状
output = model.b1(X_dummy)
print(f"Stem Output: \t\t{output.shape}")
for name, layer in model.stages.named_children():
    output = layer(output)
    print(f"{name} Output: \t\t{output.shape}")
output = model.avgpool(output)
print(f"AvgPool Output: \t{output.shape}")
output = model.flatten(output)
print(f"Flatten Output: \t{output.shape}")
output = model.fc(output)
print(f"FC Output: \t\t{output.shape}\n")

# 6. 训练设置
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 7. 训练与测试函数 (保持不变)
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 前向
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 50 == 0:  # 稍微减少打印频率
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"   [Batch] Loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / num_batches
    avg_acc = correct / size
    return avg_loss, avg_acc


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches
    avg_acc = correct / size
    return avg_loss, avg_acc


# 8. 开始训练
epochs = 5  # ResNet 训练较慢，这里演示设为 5，实际可设为 10
print(f"Start Training ResNet-18 for {epochs} epochs...")
start_time = time.time()

for t in range(epochs):
    print(f"\nEpoch {t + 1}/{epochs}")
    print("-" * 30)

    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test(test_dataloader, model, loss_fn)

    print("-" * 30)
    print(f"End of Epoch {t + 1}:")
    print(f" > Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
    print(f" > Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc * 100:.2f}%")

total_time = time.time() - start_time
print(f"\nTraining Done! Total Time: {total_time:.2f}s")

# 保存模型
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir,"ResNet_fashion_mninst.pth")
torch.save(model.state_dict(),save_path)
pritn(f"model saved to {save_path}")
