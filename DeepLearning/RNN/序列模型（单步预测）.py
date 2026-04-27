#马尔可夫假设 （另一种是潜变量模型）

#单步预测
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 1. 超参数设置
LR = 0.01  # 学习率
BATCH_SIZE = 64  # 批大小
EPOCHS = 200  # 训练轮数
TAU = 4  # 嵌入维度 (Embedding Dimension)，即马尔科夫假设的“过去tau个时刻”
DATA_NUM = 1000  # 总数据量


# 2. 数据生成
def generate_data():
    # 生成时间序列
    t = torch.linspace(0, 20, DATA_NUM)
    # x = sin(t) + 噪声
    x = torch.sin(t) + torch.normal(0, 0.1, size=(DATA_NUM,))
    return t, x


# 3. 核心：构建基于马尔科夫假设的数据集 (Sliding Window)
def create_dataset(data, tau):
    features = []
    labels = []
    # 通过滑动窗口，将序列转为 (特征, 标签) 对
    # 特征: [x_{t-tau}, ..., x_{t-1}], 标签: x_t
    for i in range(len(data) - tau):
        features.append(data[i: i + tau])
        labels.append(data[i + tau])

    # 转为Tensor
    features = torch.stack(features)
    labels = torch.stack(labels).reshape(-1, 1)  # 标签维度 (N, 1)
    return features, labels


# 准备数据
t, x = generate_data()
features, labels = create_dataset(x, TAU)

# 划分训练集和测试集 (前800个做训练，后200个做测试)
n_train = 800
train_features, test_features = features[:n_train], features[n_train:]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 放入 DataLoader
train_dataset = Data.TensorDataset(train_features, train_labels)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 4. 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  # 输入层 -> 隐藏层1
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 16),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Linear(16, 1)  # 隐藏层2 -> 输出层 (预测标量)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim=TAU)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.MSELoss()  # 回归问题通常使用均方误差

# 5. 训练循环
print("开始训练...")
loss_history = []
for epoch in range(EPOCHS):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = model(b_x)  # 前向传播
        loss = loss_func(output, b_y)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    loss_history.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# 6. 预测与可视化
model.eval()  # 切换到评估模式

# 单步预测 (One-step-ahead Prediction)
# 这里我们使用真实的测试集特征（包含真实的过去历史）来预测下一个点
with torch.no_grad():
    test_preds = model(test_features)

# 绘图
plt.figure(figsize=(12, 6))

# 绘制整体真实数据
plt.plot(t.numpy(), x.numpy(), label='Ground Truth (Data)', alpha=0.5, color='gray')

# 绘制训练部分的拟合 (可选)
with torch.no_grad():
    train_preds = model(train_features)
# 注意：预测值的x坐标需要偏移 tau
plt.plot(t[TAU:n_train + TAU].numpy(), train_preds.numpy(), label='Train Fitting', linestyle='--')

# 绘制测试部分的预测
plt.plot(t[n_train + TAU:].numpy(), test_preds.numpy(), label='Test Prediction', color='red', linewidth=2)

plt.title(f'MLP Prediction with Markov Assumption (tau={TAU})')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失曲线
plt.figure()
plt.plot(loss_history)
plt.title('Training Loss')
plt.show()