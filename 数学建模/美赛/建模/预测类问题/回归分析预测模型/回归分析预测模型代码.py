import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RegressionPredictor:
    def __init__(self, polynomial_degree=1):
        """
        :param polynomial_degree: 多项式阶数。
                                  1 = 线性回归 (y = ax + b)
                                  2 = 二次回归 (y = ax^2 + bx + c)
        """
        self.degree = polynomial_degree
        self.model = None
        self.results = None
        self.poly = None

    def fit(self, X, y):
        """
        训练模型
        :param X: 自变量 (n_samples, n_features) 或 (n_samples,)
        :param y: 因变量 (n_samples,)
        """
        # 整理 X 的形状
        X_in = np.array(X)
        if X_in.ndim == 1:
            X_in = X_in.reshape(-1, 1)

        y_in = np.array(y)

        # 1. 处理多项式特征 (如果 degree > 1)
        if self.degree > 1:
            self.poly = PolynomialFeatures(degree=self.degree)
            X_transformed = self.poly.fit_transform(X_in)
        else:
            X_transformed = X_in
            # statsmodels 需要手动添加截距项 (Constant/Intercept)
            X_transformed = sm.add_constant(X_transformed)

        # 2. 最小二乘法 (OLS) 拟合
        self.model = sm.OLS(y_in, X_transformed)
        self.results = self.model.fit()

        # 3. 打印详细的统计报告 (直接截图放论文附录)
        print("-" * 30)
        print(f"回归分析报告 (阶数={self.degree}):")
        print(self.results.summary())
        print("-" * 30)

    def predict(self, X_new):
        """
        预测
        """
        X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        if self.degree > 1:
            X_transformed = self.poly.transform(X_new)
        else:
            X_transformed = sm.add_constant(X_new, has_constant='add')

        return self.results.predict(X_transformed)

    def plot(self, X, y):
        """
        绘图展示 (仅支持单变量 X 的可视化，多变量X画不出图)
        """
        X_arr = np.array(X)
        if X_arr.ndim > 1 and X_arr.shape[1] > 1:
            print("提示: 自变量多于1个，无法绘制二维拟合曲线图，仅绘制 真实值 vs 预测值 图。")
            y_pred = self.predict(X)
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, color='blue')
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('真实值 vs 预测值')
            plt.grid(True)
            plt.show()
            return

        # 单变量绘图逻辑
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='black', label='真实数据')

        # 生成平滑曲线
        X_range = np.linspace(X_arr.min(), X_arr.max(), 100).reshape(-1, 1)
        y_pred_smooth = self.predict(X_range)

        plt.plot(X_range, y_pred_smooth, color='red', linewidth=2, label=f'{self.degree}阶拟合曲线')

        # 绘制置信区间 (Confidence Interval) - 高级功能
        # 获取预测结果的详细信息
        if self.degree == 1:  # 仅线性回归演示置信区间绘制
            X_smooth_const = sm.add_constant(X_range)
            predictions = self.results.get_prediction(X_smooth_const)
            pred_df = predictions.summary_frame(alpha=0.05)  # 95% CI

            plt.fill_between(X_range.flatten(),
                             pred_df['mean_ci_lower'],
                             pred_df['mean_ci_upper'],
                             color='red', alpha=0.1, label='95%置信区间')

        plt.title(f'回归分析拟合 (Degree={self.degree})')
        plt.xlabel('自变量 X')
        plt.ylabel('因变量 Y')
        plt.legend()
        plt.grid(True)
        plt.show()


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景1：简单的线性关系 (广告费 -> 销售额)
    print("=== 案例1: 线性回归 ===")
    x1 = [10, 20, 30, 40, 50, 60, 70, 80]
    y1 = [25, 45, 60, 85, 110, 135, 150, 180]

    model_lin = RegressionPredictor(polynomial_degree=1)
    model_lin.fit(x1, y1)
    model_lin.plot(x1, y1)

    # 场景2：非线性关系 (如: 施肥量 -> 产量，到了某个点会饱和下降)
    print("\n=== 案例2: 二次多项式回归 ===")
    x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 构造一个先升后降的数据
    y2 = [10, 25, 45, 60, 70, 75, 72, 60, 40, 15]

    model_poly = RegressionPredictor(polynomial_degree=2)
    model_poly.fit(x2, y2)

    # 预测一下投入为 5.5 时的情况
    pred_val = model_poly.predict([5.5])
    print(f"输入 5.5，预测输出: {pred_val[0]:.2f}")

    model_poly.plot(x2, y2)