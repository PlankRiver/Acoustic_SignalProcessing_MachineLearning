import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

batch_size = 128

# ============================================================
# 【重点修改】定义数据增广
# ============================================================

# 1. 训练集转换 (Train Transform)：包含数据增广
# 这里的逻辑是：先做几何变换(翻转、旋转)，最后再 Resize 到网络需要的输入大小
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转 (左右镜像)
    transforms.RandomRotation(degrees=15),   # 随机旋转 -15度 到 +15度
    transforms.Resize((224, 224)),           # AlexNet 需要 224x224
    transforms.ToTensor()                    # 转为 Tensor
])

# 2. 测试集转换 (Test Transform)：不做增广，只做必要处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 必须和训练集大小一致
    transforms.ToTensor()
])

# 加载数据集时，分别传入不同的 transform
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=train_transform
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=test_transform
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ============================================================
# 以下模型定义与训练代码保持不变
# ============================================================

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),nn.Flatten(),
            nn.Linear(in_features=6400, out_features=4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),nn.ReLU(),nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )
    def forward(self, x):
        return self.net(x)

model = AlexNet().to(device)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 20 == 0:
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

epochs = 10
print(f"\nStart Training with Data Augmentation for {epochs} epochs...")
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

model_path = "AlexNet_fashion_mnist_augmented.pth"
torch.save(model.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")
