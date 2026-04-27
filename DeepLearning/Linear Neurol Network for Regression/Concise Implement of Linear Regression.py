import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 生成合成数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 加入高斯噪声
    return X, y.reshape((-1, 1))

# 定义真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成数据
features, labels = synthetic_data(true_w, true_b, 1000)

# 使用 PyTorch 的 DataLoader
batch_size = 32
dataset = TensorDataset(features, labels)
data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入特征维度为2，输出维度为1
        self.net = nn.Linear(2, 1)

        # 初始化参数
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

# 实例化模型
model = LinearRegression()

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

# 4. 训练循环
num_epochs = 3
print("开始训练...")

for epoch in range(num_epochs):
    for X, y in data_iter:
        # 前向传播
        output = model(X)
        l = loss_fn(output, y)

        # 反向传播
        optimizer.zero_grad()  # 清空过往梯度
        l.backward()  # 计算梯度
        optimizer.step()  # 更新参数

    print(f'Epoch {epoch + 1}, Loss: {l.item():f}')

# 5. 结果验证 (训练后内存中的模型)
print("\n训练结束，验证结果：")
w_hat = model.net.weight.data.reshape(true_w.shape)
b_hat = model.net.bias.data
print(f'真实的 w: {true_w}')
print(f'预测的 w: {w_hat}')
print(f'error in estimating w: {true_w - w_hat}')
print(f'error in estimating b: {true_b - b_hat}')

# ==========================================
# 6. 保存模型 (新增部分)
# ==========================================
model_path = "linear_regression_model.pth"
# 保存参数字典 (state_dict)，其中包含 weight 和 bias
torch.save(model.state_dict(), model_path)
print(f"\n模型参数已保存至: {os.path.abspath(model_path)}")

# ==========================================
# 7. 加载模型并验证 (新增部分)
# ==========================================
print("\n正在加载保存的模型进行验证...")

# 1. 必须先实例化一个新的模型对象 (结构要和保存时一致)
loaded_model = LinearRegression()

# 2. 加载参数
# weights_only=True 是为了安全，防止加载恶意 pickle 文件，只加载权重数据
loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
loaded_model.eval() # 设置为评估模式 (对于简单的 Linear 层其实没区别，但在含 Dropout/BN 时很重要)

# 3. 查看加载后的参数是否正确
loaded_w = loaded_model.net.weight.data.reshape(true_w.shape)
loaded_b = loaded_model.net.bias.data

print(f"加载后的预测 w: {loaded_w}")
print(f"加载后的预测 b: {loaded_b}")

# 4. 做一个简单的预测测试
test_input = torch.tensor([1.0, -1.0]) # 假设一个输入
# 真实值应该是: 1*2 + (-1)*(-3.4) + 4.2 = 2 + 3.4 + 4.2 = 9.6
with torch.no_grad():
    prediction = loaded_model(test_input)
    print(f"输入 {test_input} 的预测结果: {prediction.item():.4f}")
    print(f"理论真实值 (不含噪声): {torch.dot(test_input, true_w) + true_b:.4f}")