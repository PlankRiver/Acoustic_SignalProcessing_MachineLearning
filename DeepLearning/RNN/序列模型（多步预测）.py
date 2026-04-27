import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 1. 超参数设置
LR = 0.01  # 学习率
BATCH_SIZE = 64  # 批大小
EPOCHS = 200  # 训练轮数
TAU = 4  # 嵌入维度 (马尔科夫假设：利用过去 tau 个点预测下一个)
DATA_NUM = 1000  # 总数据量


# 2. 数据生成
def generate_data():
    t = torch.linspace(0, 20, DATA_NUM)
    # x = sin(t) + 噪声
    x = torch.sin(t) + torch.normal(0, 0.1, size=(DATA_NUM,))
    return t, x


# 3. 数据集构建 (Sliding Window)
def create_dataset(data, tau):
    features = []
    labels = []
    for i in range(len(data) - tau):
        features.append(data[i: i + tau])
        labels.append(data[i + tau])

    features = torch.stack(features)
    labels = torch.stack(labels).reshape(-1, 1)
    return features, labels


# 准备数据
t, x = generate_data()
features, labels = create_dataset(x, TAU)

# 划分训练集和测试集
n_train = 800
train_features, test_features = features[:n_train], features[n_train:]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# DataLoader
train_dataset = Data.TensorDataset(train_features, train_labels)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 4. 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim=TAU)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.MSELoss()

# 5. 训练循环
print("开始训练...")
loss_history = []
for epoch in range(EPOCHS):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = model(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

# 6. 核心：多步预测与可视化
model.eval()

# --- A. 单步预测 (Single-step Prediction) ---
# 逻辑：每次预测都使用"真实的"过去数据作为输入 (Teacher Forcing)
# 作用：检测模型在理想情况下的拟合能力
with torch.no_grad():
    onestep_preds = model(test_features)

# --- B. 多步预测 (Multi-step / Autoregressive Prediction) ---
# 逻辑：使用"上一步的预测值"作为输入，滚动预测
# 作用：检测模型在脱离真实数据后的真实生成能力
multistep_preds = []

# 初始输入：取测试集的第一个特征窗口 (即训练集的最后 tau 个真实点)
# shape: (1, TAU)
current_input = test_features[0].unsqueeze(0)

with torch.no_grad():
    for i in range(len(test_labels)):
        # 1. 预测下一步
        pred = model(current_input)  # pred shape: (1, 1)

        # 2. 记录结果
        multistep_preds.append(pred.item())

        # 3. 更新输入窗口 (关键步骤)
        # 丢弃最旧的 x(t-tau)，在末尾拼接新的预测值 pred
        # dim=1 表示在特征维度拼接
        current_input = torch.cat((current_input[:, 1:], pred), dim=1)

# 转为Tensor方便绘图
multistep_preds = torch.tensor(multistep_preds)

# --- C. 绘图对比 ---
plt.figure(figsize=(12, 6))

# 1. 绘制真实数据 (Ground Truth)
plt.plot(t[n_train + TAU:].numpy(), test_labels.numpy(),
         label='Ground Truth', color='gray', alpha=0.5, linewidth=4)

# 2. 绘制单步预测 (Standard MLP usage)
plt.plot(t[n_train + TAU:].numpy(), onestep_preds.numpy(),
         label='1-step Prediction (Uses Real History)', linestyle='--', color='blue')

# 3. 绘制多步预测 (Autoregressive usage)
plt.plot(t[n_train + TAU:].numpy(), multistep_preds.numpy(),
         label='Multi-step Prediction (Uses Predicted History)',
         color='red', linewidth=2, marker='.', markersize=2)

plt.title(f'MLP Multi-step Prediction (tau={TAU})')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()