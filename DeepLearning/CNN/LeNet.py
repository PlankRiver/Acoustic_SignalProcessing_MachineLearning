import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
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

# 2. 准备数据
batch_size = 256
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# 3. 定义模型
class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


model = LeNet().to(device)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 因为使用 ReLU，所以推荐用 kaiming_uniform_
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model.apply(init_weights)


# 4. 优化器与损失函数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. 训练函数 (修改：统计并返回准确率)
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

        # 统计数据
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # 每 20 个 batch 打印一次进度 (心跳包)
        if batch % 20 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"   [Batch] Loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / num_batches
    avg_acc = correct / size
    return avg_loss, avg_acc


# 6. 测试函数
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


# 7. 主循环
epochs = 10
print(f"\nStart Training for {epochs} epochs...")

start_time = time.time()

for t in range(epochs):
    print(f"\nEpoch {t + 1}/{epochs}")
    print("-" * 30)

    # 训练
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)

    # 测试
    test_loss, test_acc = test(test_dataloader, model, loss_fn)

    # 【重点】详细输出结果
    print("-" * 30)
    print(f"End of Epoch {t + 1}:")
    print(f" > Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
    print(f" > Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc * 100:.2f}%")

total_time = time.time() - start_time
print(f"\nTraining Done! Total Time: {total_time:.2f}s")

# 保存模型
model_path = "LeNet_fashion_mnist_kaiming.pth"
torch.save(model.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")