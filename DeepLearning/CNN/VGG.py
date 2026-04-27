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



def VGG_blocks(conv_arch):
    def VGG_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    in_channels = 1
    conv_bulk = []
    for num_convs, out_channels in conv_arch:
        conv_bulk.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_bulk)


class VGG(nn.Module):
    def __init__(self,conv_arch):
        super().__init__()
        self.net = nn.Sequential(
            VGG_blocks(conv_arch),nn.Flatten(),
            nn.Linear(conv_arch[-1][-1]*7*7,4096),nn.ReLU(),
            nn.Dropout(0.5),nn.Linear(4096,4096),nn.ReLU(),
            nn.Dropout(0.5),nn.Linear(4096,10)
        )
    def forward(self, x):
        return self.net(x)

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

model = VGG(conv_arch).to(device)
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 因为使用 ReLU，所以推荐用 kaiming_uniform_
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
model.apply(init_weights)

#查看每一层的形状
X = torch.randn(1,1,224,224).to(device)
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
model_path = "VGG_fashion_mnist.pth"
torch.save(model.state_dict(), model_path)
print(f"Saved PyTorch Model State to {model_path}")
