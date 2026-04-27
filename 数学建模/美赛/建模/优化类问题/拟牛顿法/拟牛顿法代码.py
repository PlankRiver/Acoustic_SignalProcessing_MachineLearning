import numpy as np
from scipy.optimize import minimize


# 1. 定义目标函数
def objective(x):
    # 示例：Rosenbrock 函数 (经典的非线性优化测试函数)
    # f(x, y) = 100*(y - x^2)^2 + (1 - x)^2
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


# 2. 定义梯度函数 (如果不提供，scipy 会用差分法估算，但提供梯度会更快更准)
def derivative(x):
    df_dx = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    df_dy = 200 * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])


def solve_with_bfgs():
    # 初始点
    x0 = np.array([-1.2, 1.0])

    # 使用 BFGS 算法求解
    # jac 参数即 Jacobian (一阶导数)
    res = minimize(objective, x0, method='BFGS', jac=derivative,
                   options={'disp': True, 'gtol': 1e-6})

    if res.success:
        print("-" * 30)
        print(f"拟牛顿法求解成功！")
        print(f"最优解 x: {res.x}")
        print(f"最小值 f(x): {res.fun}")
        print(f"迭代次数: {res.nit}")
        print(f"梯度计算次数: {res.njev}")
    else:
        print("优化失败:", res.message)


# ================= 运行示例 =================

if __name__ == "__main__":
    solve_with_bfgs()