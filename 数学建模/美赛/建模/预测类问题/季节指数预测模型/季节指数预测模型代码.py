import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SeasonalPredictor:
    def __init__(self, data, period):
        """
        :param data: 一维数组或列表，按时间排序的数据
        :param period: 周期长度 (如: 季度=4, 月度=12, 周=7)
        """
        self.data = np.array(data)
        self.period = period
        self.indices = None  # 存储季节指数
        self.model_trend = None  # 存储线性回归模型
        self.trend_fit = None  # 存储拟合的趋势线

    def fit(self):
        # 1. 使用 statsmodels 进行分解 (乘法模型: observed = trend * seasonal * resid)
        # 注意: 乘法模型要求数据必须 > 0
        df = pd.DataFrame(self.data, columns=['value'])

        # decomposition 结果包含: trend, seasonal, resid
        # extrapolate_trend='freq' 表示用线性插值填充两端的 NaN
        result = seasonal_decompose(df['value'], model='multiplicative', period=self.period, extrapolate_trend='freq')

        # 2. 获取季节指数 (Seasonal Indices)
        # result.seasonal 是一个完整的序列，但其实只有 period 个唯一的指数在循环
        # 我们取前 period 个值作为基准指数
        self.indices = result.seasonal.iloc[:self.period].values

        # 3. 获取趋势项 (Trend)
        trend = result.trend.values

        # 4. 对趋势项进行线性回归拟合 (Trend Prediction)
        # X = 时间步 [0, 1, 2, ...], Y = 趋势值
        X = np.arange(len(trend)).reshape(-1, 1)
        y = trend

        self.model_trend = LinearRegression()
        self.model_trend.fit(X, y)
        self.trend_fit = self.model_trend.predict(X)

        print("-" * 30)
        print("季节指数计算完成:")
        for i, idx in enumerate(self.indices):
            print(f"时期 {i + 1}: {idx:.4f}")

    def predict(self, future_steps):
        """
        预测未来
        :param future_steps: 预测多少个时间步
        """
        n = len(self.data)

        # 1. 预测未来的趋势 (Trend)
        future_X = np.arange(n, n + future_steps).reshape(-1, 1)
        future_trend = self.model_trend.predict(future_X)

        # 2. 匹配未来的季节指数 (Seasonality)
        # 比如现在的 n=12 (刚好一年)，下一个预测点 n+1 对应的就是 index[0]
        # 用取模运算 % 来循环索引
        future_indices = []
        for i in range(future_steps):
            idx_pos = (n + i) % self.period
            future_indices.append(self.indices[idx_pos])

        future_indices = np.array(future_indices)

        # 3. 最终预测 (Trend * Seasonal)
        future_pred = future_trend * future_indices

        return future_pred

    def plot(self, future_pred):
        plt.figure(figsize=(10, 6))

        # 画历史真实数据
        plt.plot(self.data, 'o-', color='black', label='真实数据')

        # 画拟合数据 (历史趋势 * 历史季节指数)
        # 注意：这里的 fitted 仅仅是为了展示效果
        fitted_hist = self.trend_fit * np.tile(self.indices, int(np.ceil(len(self.data) / self.period)))[
                                       :len(self.data)]
        plt.plot(fitted_hist, '--', color='green', alpha=0.7, label='模型拟合 (训练集)')

        # 画预测数据
        x_future = np.arange(len(self.data), len(self.data) + len(future_pred))
        plt.plot(x_future, future_pred, 'x-', color='red', lw=2, label='未来预测')

        plt.title('季节指数预测模型 (Multiplicative Decomposition)')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.legend()
        plt.grid(True)
        plt.show()


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：某冰淇淋店过去 3 年 (12个季度) 的销量
    # 特征：每年 Q2, Q3 高，Q1, Q4 低，且整体销量在逐年增长

    # 构造数据: 趋势(10, 12...) + 季节性因子 + 随机噪音
    trend_base = np.linspace(100, 200, 12)  # 线性增长
    seasonal_factor = np.array([0.6, 1.3, 1.5, 0.6])  # 季节指数 (冬, 春, 夏, 秋) -> 假设Q2Q3热
    seasonality = np.tile(seasonal_factor, 3)  # 复制3年

    # 观测值 = 趋势 * 季节 * 随机
    y = trend_base * seasonality * np.random.normal(1, 0.05, 12)

    print("原始季度销量:", np.round(y, 1))

    # 1. 初始化 (周期=4，因为是季度数据)
    model = SeasonalPredictor(y, period=4)

    # 2. 训练
    model.fit()

    # 3. 预测未来 1 年 (4个季度)
    preds = model.predict(future_steps=4)
    print("-" * 30)
    print("未来4季度预测:", np.round(preds, 2))

    # 4. 绘图
    model.plot(preds)