import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LogisticPredictor:
    def __init__(self):
        self.popt = None  # 存储拟合的最优参数 [K, A, r]
        self.t_start = 0  # 记录起始年份，用于归一化

    def logistic_func(self, t, K, A, r):
        """
        Logistic 基础公式: y = K / (1 + A * exp(-r * t))
        """
        return K / (1 + A * np.exp(-r * t))

    def fit(self, t_data, y_data):
        """
        拟合模型
        :param t_data: 年份/时间数组
        :param y_data: 数量/数值数组
        """
        t_data = np.array(t_data)
        y_data = np.array(y_data)

        # 1. 时间归一化 (防止年份2000导致exp爆炸)
        self.t_start = t_data[0]
        t_norm = t_data - self.t_start

        # 2. 设置参数初始猜测值 (Initial Guess) - 这步很重要！
        # K (天花板): 猜它是当前最大值的2倍，或者是当前最大值
        # A (初始状态): 随便给个 1.0
        # r (增长率): 随便给个 0.1
        # 如果不给 bounds，有时候算出负数 K 就不好了
        # bounds=([0, 0, 0], [np.inf, np.inf, 5]) -> K, A, r 必须大于0

        p0 = [max(y_data) * 1.2, 1, 0.1]

        try:
            # 使用非线性最小二乘法拟合
            self.popt, pcov = curve_fit(self.logistic_func, t_norm, y_data, p0=p0, maxfev=5000)

            K_fit, A_fit, r_fit = self.popt
            print("-" * 30)
            print("模型拟合成功！")
            print(f"预测环境容纳量 (K): {K_fit:.2f}")
            print(f"初始参数 (A): {A_fit:.4f}")
            print(f"增长率 (r): {r_fit:.4f}")

            # 计算拟合优度 R^2
            y_pred = self.logistic_func(t_norm, *self.popt)
            r2 = r2_score(y_data, y_pred)
            print(f"拟合优度 R^2: {r2:.4f}")
            print("-" * 30)

        except Exception as e:
            print("拟合失败，可能数据不符合S型趋势，或初始猜测不佳。")
            print("错误信息:", e)

    def predict(self, future_years):
        """
        预测未来
        :param future_years: 待预测的年份列表
        :return: 预测值数组
        """
        if self.popt is None:
            raise ValueError("请先调用 fit() 训练模型")

        future_years = np.array(future_years)
        t_norm = future_years - self.t_start  # 记得减去起始年份
        return self.logistic_func(t_norm, *self.popt)


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：某地区互联网用户数（万人）增长预测
    # 数据呈现慢 -> 快 -> 慢的 S 型特征
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    values = [100, 150, 300, 600, 1200, 2000, 2800, 3300, 3500, 3600]

    # 1. 初始化并训练
    model = LogisticPredictor()
    model.fit(years, values)

    # 2. 预测未来 5 年
    future = [2020, 2021, 2022, 2023, 2024]
    pred_vals = model.predict(future)
    print(f"未来5年预测值: {np.round(pred_vals, 2)}")

    # 3. 绘图展示
    # 生成平滑曲线用于画图
    all_years = np.linspace(min(years), max(future), 100)
    smooth_curve = model.predict(all_years)

    plt.figure(figsize=(8, 5))
    plt.scatter(years, values, color='black', label='历史数据')
    plt.plot(all_years, smooth_curve, color='blue', label='Logistic 拟合曲线')
    plt.scatter(future, pred_vals, color='red', marker='x', s=100, label='预测值')

    # 画出 K 值虚线
    K_val = model.popt[0]
    plt.axhline(y=K_val, color='green', linestyle='--', label=f'上限 K={K_val:.0f}')

    plt.xlabel('年份')
    plt.ylabel('数量')
    plt.title('Logistic 增长模型预测')
    plt.legend()
    plt.grid(True)
    plt.show()