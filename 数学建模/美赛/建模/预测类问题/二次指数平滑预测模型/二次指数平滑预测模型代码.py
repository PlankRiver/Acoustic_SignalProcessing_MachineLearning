import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class HoltLinearTrend:
    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.params = {}
        self.level = []
        self.trend = []
        self.fitted = []  # 存储历史拟合值

    def fit(self, data):
        """
        训练模型
        :param data: 一维数组或列表
        """
        self.data = np.array(data)
        n = len(self.data)

        # 如果没有指定参数，则进行网格搜索寻找最优参数
        if self.alpha is None or self.beta is None:
            self._auto_tune()

        # 使用确定的参数重新跑一遍记录过程
        self._run_holt(self.alpha, self.beta)
        print(f"模型训练完成! 最优参数: alpha={self.alpha:.4f}, beta={self.beta:.4f}")

    def _run_holt(self, alpha, beta):
        """运行 Holt 算法的核心逻辑"""
        n = len(self.data)
        L = np.zeros(n)
        T = np.zeros(n)
        fitted = np.zeros(n)

        # 1. 初始化
        L[0] = self.data[0]
        T[0] = self.data[1] - self.data[0]
        fitted[0] = L[0]  # 第0个点的拟合值通常就是它自己

        # 2. 迭代
        for t in range(1, n):
            # 记录上一时刻的预测值 (One-step-ahead forecast)
            fitted[t] = L[t - 1] + T[t - 1]

            # 更新 Level
            L[t] = alpha * self.data[t] + (1 - alpha) * (L[t - 1] + T[t - 1])
            # 更新 Trend
            T[t] = beta * (L[t] - L[t - 1]) + (1 - beta) * T[t - 1]

        self.level = L
        self.trend = T
        self.fitted = fitted
        return fitted

    def _auto_tune(self):
        """网格搜索寻找 MSE 最小的 alpha 和 beta"""
        best_score = float('inf')
        best_params = (0.5, 0.5)

        # 步长 0.1 遍历 0.1 到 0.9
        grid = np.arange(0.1, 1.0, 0.1)

        for a in grid:
            for b in grid:
                fitted = self._run_holt(a, b)
                # 计算 MSE (从第1个点开始算，因为第0个点是初始化的)
                mse = mean_squared_error(self.data[1:], fitted[1:])
                if mse < best_score:
                    best_score = mse
                    best_params = (a, b)

        self.alpha, self.beta = best_params

    def predict(self, steps=5):
        """
        预测未来
        :param steps: 预测步数
        """
        last_L = self.level[-1]
        last_T = self.trend[-1]

        preds = []
        for m in range(1, steps + 1):
            # 预测公式: L_t + m * T_t
            pred = last_L + m * last_T
            preds.append(pred)

        return np.array(preds)


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：预测某新兴产品的销量 (呈现线性增长趋势)
    # 原始数据 (带有一点波动，但总体向上)
    history_data = [10, 12, 16, 19, 23, 26, 30, 35, 39, 44]
    years = np.arange(2010, 2020)

    print("原始数据:", history_data)

    # 1. 初始化模型 (不传参数，让它自己学)
    model = HoltLinearTrend()

    # 2. 训练
    model.fit(history_data)

    # 3. 预测未来 5 年
    future_steps = 5
    predictions = model.predict(future_steps)
    print("-" * 30)
    print("未来5年预测值:", np.round(predictions, 2))

    # 4. 绘图
    plt.figure(figsize=(8, 5))

    # 画历史数据
    plt.plot(years, history_data, 'o-', color='black', label='真实数据')

    # 画拟合数据 (看看模型对历史的拟合程度)
    # 注意：fitted[0] 通常没意义，可以从 fitted[1] 开始画
    plt.plot(years, model.fitted, '--', color='green', label='模型拟合')

    # 画预测数据
    future_years = np.arange(years[-1] + 1, years[-1] + 1 + future_steps)
    plt.plot(future_years, predictions, 'x-', color='red', label='二次指数平滑预测')

    plt.xlabel('年份')
    plt.ylabel('销量')
    plt.title('二次指数平滑 (Holt Linear Trend) 预测')
    plt.legend()
    plt.grid(True)
    plt.show()