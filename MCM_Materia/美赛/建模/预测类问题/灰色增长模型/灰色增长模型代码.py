import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GreyForecast:
    def __init__(self, data):
        """
        :param data: 一维列表或数组，原始数据
        """
        self.data = np.array(data, dtype=float)
        self.n = len(self.data)
        self.a = None  # 发展系数
        self.b = None  # 灰作用量

    def fit(self):
        """模型训练"""
        # 1. 级比检验 (这里只做打印提示，不强制阻断，实际比赛中可以在文档中说明)
        lambdas = self.data[:-1] / self.data[1:]
        min_range = np.exp(-2 / (self.n + 1))
        max_range = np.exp(2 / (self.n + 1))
        print(f"级比检验范围: ({min_range:.4f}, {max_range:.4f})")
        # 如果超出范围，建议对 self.data 加上一个常数 c 再进行训练

        # 2. 累加生成 (AGO)
        x1 = self.data.cumsum()

        # 3. 构造紧邻均值序列
        z1 = (x1[:-1] + x1[1:]) / 2.0

        # 4. 构造矩阵 B 和 Y
        B = np.zeros((self.n - 1, 2))
        B[:, 0] = -z1
        B[:, 1] = 1

        Y = self.data[1:].reshape(-1, 1)

        # 5. 最小二乘法求解 a, b
        # (B.T * B)^-1 * B.T * Y
        # 使用 numpy 的 lstsq 更稳定
        params, residuals, rank, s = np.linalg.lstsq(B, Y, rcond=None)
        self.a = params[0][0]
        self.b = params[1][0]

        print(f"参数解算完成: a = {self.a:.4f}, b = {self.b:.4f}")

    def predict(self, future_steps=5):
        """
        预测未来
        :param future_steps: 向后预测多少步
        :return: 包含历史拟合值和未来预测值的完整数组
        """
        if self.a is None:
            self.fit()

        # 预测公式: x(1)(k+1) = (x(0)(1) - b/a) * exp(-ak) + b/a
        # 注意: k 从 0 开始

        preds_x1 = []
        # 计算的总长度 = 历史长度 + 未来长度
        total_len = self.n + future_steps

        # x(1) 的预测值
        for k in range(total_len):
            val = (self.data[0] - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
            preds_x1.append(val)

        preds_x1 = np.array(preds_x1)

        # 累减还原: x(0)(k) = x(1)(k) - x(1)(k-1)
        # 第一个值直接是 x(0)(1)
        preds_x0 = np.zeros(total_len)
        preds_x0[0] = self.data[0]

        for k in range(1, total_len):
            preds_x0[k] = preds_x1[k] - preds_x1[k - 1]

        return preds_x0

    def evaluate(self):
        """
        后验差检验 (Model Accuracy Check)
        计算 C 值 (后验差比) 和 P 值 (小误差概率)
        """
        # 只取历史部分的拟合值
        history_pred = self.predict(future_steps=0)

        # 残差
        residuals = self.data - history_pred

        # 原始数据方差 S1
        s1_sq = np.var(self.data, ddof=1)  # ddof=1 为样本方差

        # 残差方差 S2
        s2_sq = np.var(residuals, ddof=1)

        # 后验差比 C
        C = np.sqrt(s2_sq) / np.sqrt(s1_sq)

        # 小误差概率 P
        # P = P(|e(k) - mean(e)| < 0.6745 * S1)
        mean_residual = np.mean(residuals)
        threshold = 0.6745 * np.sqrt(s1_sq)
        p_count = np.sum(np.abs(residuals - mean_residual) < threshold)
        P = p_count / self.n

        print("-" * 30)
        print(f"模型精度检验:")
        print(f"后验差比 C = {C:.4f} (越小越好, <0.35为优, <0.5为合格)")
        print(f"小误差概率 P = {P:.4f} (越大越好, >0.95为优, >0.8为合格)")

        if C < 0.35 and P > 0.95:
            print(">> 评价: 模型精度等级 [好]")
        elif C < 0.5 and P > 0.8:
            print(">> 评价: 模型精度等级 [合格]")
        else:
            print(">> 评价: 模型精度等级 [不合格] -> 建议对数据做平移变换或使用其他模型")

        return C, P


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：预测某城市未来几年的污水排放量
    # 历史数据 (2015-2020)
    history_data = [174, 179, 183, 189, 207, 234]
    years = np.arange(2015, 2021)

    print("原始数据:", history_data)

    # 1. 初始化模型
    gm = GreyForecast(history_data)

    # 2. 训练
    gm.fit()

    # 3. 精度检验
    gm.evaluate()

    # 4. 预测未来 5 年
    future_steps = 5
    prediction = gm.predict(future_steps)

    print("-" * 30)
    print("预测结果:", np.round(prediction, 2))

    # 5. 绘图
    plt.figure(figsize=(8, 5))

    # 画历史数据
    plt.plot(years, history_data, 'o-', label='历史真实值')

    # 画预测数据 (包含对历史的拟合 + 未来的预测)
    future_years = np.arange(2015, 2021 + future_steps)
    plt.plot(future_years, prediction, 'x--', color='red', label='GM(1,1) 预测值')

    plt.xlabel('年份')
    plt.ylabel('数值')
    plt.title('GM(1,1) 灰色预测')
    plt.legend()
    plt.grid(True)
    plt.show()