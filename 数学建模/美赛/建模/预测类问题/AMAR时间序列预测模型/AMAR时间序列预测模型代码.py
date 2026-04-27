import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# 忽略数值计算中的警告
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ArimaPredictor:
    def __init__(self, data):
        """
        :param data: 一维时间序列数据 (list 或 numpy array)
        """
        self.data = pd.Series(data)
        self.best_order = None  # (p, d, q)
        self.model = None

    def check_stationarity(self):
        """
        ADF 检验数据平稳性
        :return: 建议的差分阶数 d
        """
        # 0阶差分检验
        adf_res = adfuller(self.data)
        p_value = adf_res[1]
        print(f"原始数据 ADF p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(">> 数据平稳，建议 d=0")
            return 0

        # 1阶差分检验
        diff1 = self.data.diff().dropna()
        adf_res1 = adfuller(diff1)
        p_value1 = adf_res1[1]
        print(f"一阶差分 ADF p-value: {p_value1:.4f}")

        if p_value1 < 0.05:
            print(">> 一阶差分后平稳，建议 d=1")
            return 1
        else:
            print(">> 一阶差分仍不平稳，建议 d=2")
            return 2

    def auto_fit(self, max_p=3, max_q=3):
        """
        网格搜索寻找最优 (p, d, q) 组合，依据 AIC 准则
        """
        # 1. 确定 d
        d = self.check_stationarity()

        best_aic = float('inf')
        best_cfg = None

        print("-" * 30)
        print(f"开始寻找最优 p, q (d={d})...")

        # 2. 遍历 p 和 q
        # 注意：p和q一般不会很大，0到3通常足够
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(self.data, order=(p, d, q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_cfg = (p, d, q)
                        self.model = res
                except:
                    continue

        self.best_order = best_cfg
        print(f"最优参数找到: (p, d, q) = {self.best_order}, AIC = {best_aic:.2f}")
        print("-" * 30)

    def predict(self, steps=5):
        """
        预测未来
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 auto_fit()")

        # forecast 返回的是未来 steps 步的预测
        forecast_res = self.model.get_forecast(steps=steps)
        pred_values = forecast_res.predicted_mean

        # 获取置信区间 (默认95%)
        conf_int = forecast_res.conf_int(alpha=0.05)

        return pred_values, conf_int

    def plot_results(self, steps=5):
        """
        绘图展示
        """
        preds, conf = self.predict(steps)

        plt.figure(figsize=(10, 6))

        # 画历史数据
        plt.plot(self.data.index, self.data, label='历史数据')

        # 画预测数据
        future_index = range(len(self.data), len(self.data) + steps)
        plt.plot(future_index, preds, color='red', marker='o', label='未来预测')

        # 画置信区间 (阴影部分)
        plt.fill_between(future_index,
                         conf.iloc[:, 0],
                         conf.iloc[:, 1],
                         color='pink', alpha=0.3, label='95%置信区间')

        plt.title(f'ARIMA{self.best_order} 时间序列预测')
        plt.legend()
        plt.grid(True)
        plt.show()


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：模拟某股票收盘价 (带一点趋势和随机波动)
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    # 构造数据: 线性趋势 + 周期 + 随机噪声
    y = 2 * x + 3 * np.sin(x) + np.random.normal(0, 1, 50)

    # 1. 初始化模型
    arima = ArimaPredictor(y)

    # 2. 自动训练 (自动定阶)
    arima.auto_fit(max_p=4, max_q=4)

    # 3. 预测未来 5 个时间点
    future_preds, conf_interval = arima.predict(steps=5)

    print("未来5期预测值:")
    print(future_preds)

    # 4. 绘图
    arima.plot_results(steps=10)