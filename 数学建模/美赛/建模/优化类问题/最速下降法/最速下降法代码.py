import numpy as np


def objective_func(x):
    """目标函数 f(x, y) = x^2 + 10y^2"""
    return x[0] ** 2 + 10 * x[1] ** 2


def gradient_func(x):
    """梯度的解析表达式"""
    return np.array([2 * x[0], 20 * x[1]])


def steepest_descent(x0, eps=1e-6, max_iter=1000):
    """
    最速下降法实现
    :param x0: 初始点
    :param eps: 梯度模长的终止阈值
    :param max_iter: 最大迭代次数
    """
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):
        grad = gradient_func(x)

        # 1. 检查终止条件
        if np.linalg.norm(grad) < eps:
            print(f"算法收敛于第 {k} 次迭代")
            break

        # 2. 确定搜索方向 (负梯度)
        d = -grad

        # 3. 计算最优步长 alpha (针对二次型问题的精确解)
        # 对于 f = 1/2 * x.T * Q * x, alpha = (g.T * g) / (g.T * Q * g)
        # 这里为了通用，可以简写，针对本题 Q = [[2, 0], [0, 20]]
        Q = np.array([[2, 0], [0, 20]])
        alpha = np.dot(grad, grad) / np.dot(np.dot(grad, Q), grad)

        # 4. 更新坐标
        x = x + alpha * d
        path.append(x.copy())
    else:
        print("达到最大迭代次数")

    return x, np.array(path)


# ================= 使用示例 =================

if __name__ == "__main__":
    # 初始点设为 (10, 1)
    start_point = [10, 1]

    optimal_x, history = steepest_descent(start_point)

    print("-" * 30)
    print(f"最优解: {optimal_x}")
    print(f"最小值: {objective_func(optimal_x)}")

    # 观察前5步的路径 (体现锯齿现象)
    print("前5步迭代路径:")
    print(history[:5])

    # 绘图展示 (如果安装了 matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        # 绘制等高线
        X, Y = np.meshgrid(np.linspace(-11, 11, 100), np.linspace(-2, 2, 100))
        Z = X ** 2 + 10 * Y ** 2
        plt.contour(X, Y, Z, levels=20)
        # 绘制路径
        plt.plot(history[:, 0], history[:, 1], 'ro-')
        plt.title("Steepest Descent Path (Zig-zag)")
        plt.show()
    except ImportError:
        pass