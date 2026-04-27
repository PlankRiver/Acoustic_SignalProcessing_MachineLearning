import numpy as np
import matplotlib.pyplot as plt


# 1. 定义目标函数 (能量函数)
def objective_function(x):
    return x ** 2 + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)


def simulated_annealing():
    # --- 1. 参数设置 ---
    T0 = 100.0  # 初始温度
    T_min = 1e-4  # 终止温度
    alpha = 0.98  # 降温系数 (冷却速率)
    max_iter = 100  # 每个温度下的迭代次数 (马尔可夫链长度)

    # 搜索空间 [-10, 10]
    x_current = np.random.uniform(-10, 10)
    f_current = objective_function(x_current)

    x_best = x_current
    f_best = f_current

    T = T0
    history = []

    # --- 2. 模拟退火过程 ---
    while T > T_min:
        for _ in range(max_iter):
            # (1) 产生新解 (扰动)
            # 扰动步长应随温度降低而减小，以提高局部搜索精度
            step = 0.5 * T / T0 + 0.1
            x_new = x_current + np.random.uniform(-step, step)
            x_new = np.clip(x_new, -10, 10)  # 边界约束

            f_new = objective_function(x_new)
            delta_f = f_new - f_current

            # (2) Metropolis 准则判定
            if delta_f < 0:  # 接受好解
                x_current, f_current = x_new, f_new
            else:
                # 以概率接受差解
                p = np.exp(-delta_f / T)
                if np.random.rand() < p:
                    x_current, f_current = x_new, f_new

            # (3) 更新全局最优记录
            if f_current < f_best:
                x_best, f_best = x_current, f_current

        history.append(f_best)
        T *= alpha  # 降温

    return x_best, f_best, history


# ================= 运行示例 =================

if __name__ == "__main__":
    best_x, best_f, history = simulated_annealing()

    print("-" * 30)
    print(f"全局最优解 x: {best_x:.6f}")
    print(f"最小函数值 f(x): {best_f:.6f}")

    # 绘制函数图像和最优解
    x_range = np.linspace(-10, 10, 1000)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_range, objective_function(x_range), label='目标函数')
    plt.plot(best_x, best_f, 'ro', label='找到的最优解')
    plt.title("函数寻优展示")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.title("收敛过程 (Best Fitness)")
    plt.xlabel("降温周期")
    plt.ylabel("能量 (Energy)")

    plt.tight_layout()
    plt.show()