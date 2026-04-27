import numpy as np
import matplotlib.pyplot as plt

# 1. 构造模拟数据 (y = 3x + 4 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


def sgd_linear_regression(X, y, learning_rate=0.01, epochs=50):
    """
    随机梯度下降实现线性回归
    """
    m, n = X.shape
    # 初始化权重 w 和偏置 b (n个特征+1个偏置)
    w = np.random.randn(n, 1)
    b = np.random.randn(1, 1)

    loss_history = []

    # 模拟学习率衰减 (Learning Rate Schedule)
    def lr_schedule(t):
        return learning_rate / (1 + 0.1 * t)

    for epoch in range(epochs):
        # 每一轮开始前打乱数据
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(m):
            xi = X_shuffled[i:i + 1]
            yi = y_shuffled[i:i + 1]

            # 1. 计算单个样本预测值
            y_pred = xi.dot(w) + b

            # 2. 计算该样本的梯度 (均方误差 MSE 导数)
            # L = (w*xi + b - yi)^2
            # dL/dw = 2 * (w*xi + b - yi) * xi
            # dL/db = 2 * (w*xi + b - yi)
            grad_w = 2 * xi.T.dot(y_pred - yi)
            grad_b = 2 * (y_pred - yi)

            # 3. 更新参数
            lr = lr_schedule(epoch * m + i)
            w -= lr * grad_w
            b -= lr * grad_b

        # 记录本轮结束时的总损失
        current_loss = np.mean((X.dot(w) + b - y) ** 2)
        loss_history.append(current_loss)

    return w, b, loss_history


# ================= 运行与展示 =================

w_final, b_final, history = sgd_linear_regression(X, y)

print(f"拟合结果: y = {w_final[0][0]:.4f}x + {b_final[0][0]:.4f}")
print(f"理论结果: y = 3x + 4")

# 绘图展示
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, X * w_final + b_final, color='red', label='SGD Fit')
plt.title("Linear Regression with SGD")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history)
plt.title("Loss History (震荡下行)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()