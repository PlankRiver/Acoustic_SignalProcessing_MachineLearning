import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GPRPredictor:
    def __init__(self):
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train):
        """
        训练模型
        :param X_train: 二维数组 (n_samples, n_features)
        :param y_train: 一维数组 (n_samples,)
        """
        # 1. 数据标准化 (GPR 对数值范围极其敏感)
        X_train = np.array(X_train).reshape(-1, 1)  # 假设是单变量，如果是多变量去掉reshape
        y_train = np.array(y_train).reshape(-1, 1)

        self.X_train_scaled = self.scaler_x.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train)

        # 2. 定义核函数 (Kernel)
        # ConstantKernel: 调整整体振幅
        # RBF: 拟合平滑曲线 (length_scale是初始猜测值)
        # WhiteKernel: 处理噪声 (非常重要！否则模型会强行穿过所有点，导致过拟合)
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

        # 3. 初始化并训练 GPR
        # n_restarts_optimizer: 随机重启次数，防止陷入局部最优
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0)
        self.model.fit(self.X_train_scaled, self.y_train_scaled)

        print("-" * 30)
        print("GPR 模型训练完成")
        print(f"最优核参数: {self.model.kernel_}")

    def predict(self, X_pred):
        """
        预测
        :return: 预测均值, 预测标准差 (用于画置信区间)
        """
        X_pred = np.array(X_pred).reshape(-1, 1)
        X_pred_scaled = self.scaler_x.transform(X_pred)

        # return_std=True 表示同时返回标准差
        y_mean_scaled, y_std_scaled = self.model.predict(X_pred_scaled, return_std=True)

        # 还原数据
        y_mean = self.scaler_y.inverse_transform(y_mean_scaled).flatten()

        # 标准差还原 (std 只需要乘缩放比例)
        y_std = y_std_scaled * self.scaler_y.scale_[0]

        return y_mean, y_std

    def plot(self, X_train, y_train, X_pred, y_mean, y_std):
        plt.figure(figsize=(10, 6))

        # 1. 画训练数据
        plt.scatter(X_train, y_train, c='red', marker='x', label='观测数据 (训练集)')

        # 2. 画预测均值曲线
        plt.plot(X_pred, y_mean, color='blue', label='GPR 预测均值')

        # 3. 画 95% 置信区间 (Mean ± 1.96 * Std)
        plt.fill_between(X_pred,
                         y_mean - 1.96 * y_std,
                         y_mean + 1.96 * y_std,
                         color='blue', alpha=0.2, label='95% 置信区间')

        plt.title('高斯过程回归 (GPR) 预测结果')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：预测某种稀有金属的物理特性 (数据很少，且有噪声)

    # 构造数据: X是温度，Y是强度
    # 真实函数: y = x * sin(x)
    np.random.seed(42)
    X = np.linspace(0, 10, 15)  # 只有15个样本！
    y = X * np.sin(X) + np.random.normal(0, 0.5, 15)  # 加上噪声

    # 1. 初始化模型
    gpr = GPRPredictor()

    # 2. 训练
    gpr.fit(X, y)

    # 3. 预测 (想看更密集的点，比如 0 到 12，看看外推效果)
    X_future = np.linspace(0, 12, 100)
    y_pred, y_std = gpr.predict(X_future)

    print("-" * 30)
    # 取前5个预测值展示
    print("预测均值 (前5个):", np.round(y_pred[:5], 2))
    print("不确定性 (前5个):", np.round(y_std[:5], 2))

    # 4. 绘图 (GPR最强大的地方在于那层阴影)
    gpr.plot(X, y, X_future, y_pred, y_std)