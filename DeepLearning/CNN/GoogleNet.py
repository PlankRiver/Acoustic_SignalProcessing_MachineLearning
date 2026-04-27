import torch
from torch import nn
from torch.nn import functional as F
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
# 修正数据预处理，增加 Resize
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

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels,c1,1)
        self.p2_1 = nn.Conv2d(in_channels,c2[0],1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],3,padding=1)
        self.p3_1 = nn.Conv2d(in_channels,c3[0],1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],5,padding=2)
        self.p4_1 = nn.MaxPool2d(3,1,1)
        self.p4_2 = nn.Conv2d(in_channels,c4,1)
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),1)
class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64,192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(192,64, (96, 128), (16, 32), 32),
            Inception(256,128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(480,192, (96, 208), (16, 48), 64),
            Inception(512,160, (112, 224), (24, 64), 64),
            Inception(512,128, (128, 256), (24, 64), 64),
            Inception(512,112, (144, 288), (32, 64), 64),
            Inception(528,256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(832,256, (160, 320), (32, 128), 128),
            Inception(832,384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.net = nn.Sequential(self.b1,self.b2,self.b3,self.b4,self.b5,nn.Linear(1024,10))
    def forward(self, x):
        return self.net(x)



model = GoogleNet().to(device)
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 因为使用 ReLU，所以推荐用 kaiming_uniform_
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

#查看每一层的形状
X = torch.randn(1,1,96,96).to(device)
for layer in model.net.children():
    X = layer(X)
    print(layer.__class__.__name__,'Output Shape:\t',X.shape)

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
model_path = "GoogleNet_fashion_mnist.pth"
torch.save(model.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")
