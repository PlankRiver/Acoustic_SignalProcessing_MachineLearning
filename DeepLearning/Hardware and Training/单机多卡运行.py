import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time
import os

# 1. 设置设备
# 注意：DataParallel 主要用于 CUDA 环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【改动1】打印显卡信息
if device.type == 'cuda':
    print(f"Using {torch.cuda.device_count()} CUDA devices.")
else:
    print(f"Using {device} device.")

# 2. 数据准备
# 【改动2】多卡训练时，通常建议增大 Batch Size
# 假设你有 2 张卡，128 可能太小导致显卡利用率不高，这里尝试改为 256
batch_size = 256

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

# 【改动3】num_workers 设置
# 多GPU计算很快，如果 CPU 读数据只有一个进程(num_workers=0)，显卡会一直等待 CPU。
# 建议设置为 min(CPU核心数, 4 * GPU数量)
num_workers = 4 if device.type == 'cuda' else 0

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# -----------------------------------------------------------------
# 3. 定义 ResNet 模型组件 (保持不变)
# -----------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
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
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            in_ch = input_channels if (i == 0) else num_channels
            use_1x1 = (in_ch != num_channels)
            blk.append(Residual(in_ch, num_channels, use_1x1conv=use_1x1, strides=1))
    return nn.Sequential(*blk)

class ResNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stages = nn.Sequential()
        in_channels = 64
        for i, (num_residuals, num_channels) in enumerate(arch):
            first_block = (i == 0)
            self.stages.add_module(f"stage_{i + 1}",
                                   resnet_block(in_channels, num_channels, num_residuals, first_block)
                                   )
            in_channels = num_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------

# 4. 实例化模型并启用 DataParallel
resnet18_arch = ((2, 64), (2, 128), (2, 256), (2, 512))
model = ResNet(resnet18_arch) # 先实例化，不要急着 .to(device)

# 初始化权重 (在包装之前初始化是个好习惯)
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

# 【改动4】将模型放入 GPU 并启用并行
# 1. 先把模型放到主 GPU
model = model.to(device)

# 2. 检查是否有多个 GPU，如果有，使用 DataParallel 包装
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    # 这行代码会自动把 Batch 切分到所有可见的 GPU 上
    model = nn.DataParallel(model)


# 5. 训练设置
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 训练与测试函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # DataParallel 会自动在这里把 X 拆分
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward() # Loss backward 会自动在各 GPU 汇总梯度
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 20 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            # 注意：这里的 loss.item() 是平均后的 loss
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

# 7. 开始训练
epochs = 5
print(f"Start Training ResNet-18 on {device} for {epochs} epochs...")
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

# 8. 保存模型
model_path = "ResNet18_fashion_mnist.pth"

# 【改动5】保存模型时的特殊处理
# 如果模型被 DataParallel 包装了，参数在 model.module 里
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), model_path)
    print("Saved DataParallel Model (module unwrapped)")
else:
    torch.save(model.state_dict(), model_path)
    print("Saved Standard Model")

print(f"Saved PyTorch Model State to {model_path}")
