import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 1. 辅助函数：计算 L2 范数 (仅用于最后打印验证，不参与训练逻辑)
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 2. 数据准备 (替代 Data 类)
def generate_data(num_train, num_val, num_inputs, batch_size):
    n = num_train + num_val
    # 生成特征 X
    X = torch.randn(n, num_inputs)

    # 生成真实的噪音
    noise = torch.randn(n, 1) * 0.01

    # 生成真实的 w 和 b (Ground Truth)
    true_w = torch.ones((num_inputs, 1)) * 0.01
    true_b = 0.05

    # 生成标签 y
    y = torch.matmul(X, true_w) + true_b + noise

    # 切分训练集和验证集
    X_train, X_val = X[:num_train], X[num_train:]
    y_train, y_val = y[:num_train], y[num_train:]

    # 封装为 DataLoader
    train_iter = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(TensorDataset(X_val, y_val),
                          batch_size=batch_size, shuffle=False)

    return train_iter, val_iter


# 设置超参数
num_inputs = 200
num_train = 20
num_val = 100
batch_size = 5
wd = 3  # 权重衰减系数 (weight decay)
lr = 0.01  # 学习率
epochs = 10

# 生成数据
train_loader, val_loader = generate_data(num_train, num_val, num_inputs, batch_size)


# 3. 定义模型 (替代 d2l.LinearRegression)
class LinearRegression(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        # 初始化参数 (可选，模拟 d2l 默认行为，也可以用 PyTorch 默认初始化)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(num_inputs)

# 4. 定义损失函数
loss_fn = nn.MSELoss()

# 5. 定义优化器 (核心：配置 Weight Decay)
# 对应原代码中的 configure_optimizers
# 我们只对 weight 应用 weight_decay，不对 bias 应用
optimizer = torch.optim.SGD([
    {'params': model.linear.weight, 'weight_decay': wd},
    {'params': model.linear.bias, 'weight_decay': 0.0}  # 偏置通常不进行正则化
], lr=lr)

# 6. 训练循环 (替代 trainer.fit)
print(f"Start training with weight_decay={wd}...")

for epoch in range(epochs):
    model.train()
    for X, y in train_loader:
        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # (可选) 在验证集上评估
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            pred_val = model(X_val)
            val_loss += loss_fn(pred_val, y_val).item()
    val_loss /= len(val_loader)

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.6f}")

# 7. 结果验证
print("-" * 30)
# 获取训练后的权重
final_w = model.linear.weight.data
# 计算 L2 范数
l2_norm = l2_penalty(final_w).item()

print(f'L2 norm of w: {l2_norm}')