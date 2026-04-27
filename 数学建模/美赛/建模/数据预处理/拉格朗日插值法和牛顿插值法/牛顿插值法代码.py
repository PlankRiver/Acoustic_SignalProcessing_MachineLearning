import numpy as np


def newton_interpolation(x_known, y_known, x_predict):
    """
    牛顿插值手动实现
    """
    n = len(x_known)
    # 初始化差商表 (n x n 矩阵)
    coef = np.zeros([n, n])
    coef[:, 0] = y_known

    # 计算差商表
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_known[i + j] - x_known[i])

    # 获取差商系数 (矩阵的第一行)
    a = coef[0, :]

    # 构建多项式并计算 x_predict
    res = a[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_predict - x_known[i - 1])
        res += a[i] * product
    return res


# --- 示例 ---
x = np.array([1, 2, 3, 5])
y = np.array([10, 15, 12, 18])
print(f"牛顿插值预测值: {newton_interpolation(x, y, 4):.4f}")