import numpy as np
from scipy.interpolate import lagrange


def lagrange_imputation(x_known, y_known, x_missing):
    """
    拉格朗日插值补全缺失值
    :param x_known: 已知数据的索引(坐标)
    :param y_known: 已知数据的值
    :param x_missing: 缺失数据的索引(待补全点)
    :return: 补全后的值
    """
    # 构造拉格朗日多项式对象
    poly = lagrange(x_known, y_known)

    # 预测缺失值
    # poly.coef 获取多项式系数
    return poly(x_missing)


# --- 示例 ---
x = np.array([1, 2, 3, 5])  # 索引为4的点缺失
y = np.array([10, 15, 12, 18])
x_miss = 4

val = lagrange_imputation(x, y, x_miss)
print(f"拉格朗日插值预测索引 {x_miss} 的值为: {val:.4f}")