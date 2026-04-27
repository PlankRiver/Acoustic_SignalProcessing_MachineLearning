import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

# 下载并加载数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# 3. 定义带 Dropout 的 MLP 模型
class DropoutMLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            # --- 第一层 ---
            nn.Linear(num_inputs, num_hiddens_1),
            nn.ReLU(),
            # 在第一层全连接后添加 Dropout
            nn.Dropout(dropout_1),

            # --- 第二层 ---
            nn.Linear(num_hiddens_1, num_hiddens_2),
            nn.ReLU(),
            # 在第二层全连接后添加 Dropout
            nn.Dropout(dropout_2),

            # --- 输出层 ---
            nn.Linear(num_hiddens_2, num_outputs)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


# 定义超参数
hparams = {
    'num_inputs': 784,  # FashionMNIST 28x28
    'num_outputs': 10,
    'num_hiddens_1': 256,
    'num_hiddens_2': 256,
    'dropout_1': 0.5,
    'dropout_2': 0.5,
    'lr': 0.1
}

# 实例化模型
model = DropoutMLP(**{k: v for k, v in hparams.items() if k != 'lr'}).to(device)

# 4. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hparams['lr'])


# 5. 训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 【非常重要】启用 Dropout
    model.train()

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 6. 测试函数
def test(dataloader, model, loss_fn):
    # 【非常重要】禁用 Dropout，确保测试结果稳定
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 7. 开始训练
epochs = 10
print(f"Start Training with Dropout (p1={hparams['dropout_1']}, p2={hparams['dropout_2']})...")

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")