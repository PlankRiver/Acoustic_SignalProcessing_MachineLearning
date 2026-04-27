import numpy as np
from scipy.optimize import minimize


def solve_nonlinear_programming():
    """
    非线性规划求解模板

    【题目示例】：
    目标：Min f(x) = x1^2 + x2^2 + x3^2 + 8

    约束条件：
    1. x1^2 - x2 + x3^2 >= 0      (不等式约束)
    2. x1 + x2^2 + x3^3 = 20      (等式约束)
    3. x1, x2, x3 >= 0            (变量边界)
    """

    # --- 1. 定义目标函数 ---
    def objective(x):
        # x 是一个数组，x[0]代表x1, x1[1]代表x2...
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 8

    # --- 2. 定义约束条件 ---
    # Scipy 规定：
    # 'type': 'ineq' 代表不等式约束 (默认形式是 函数值 >= 0)
    # 'type': 'eq'   代表等式约束   (默认形式是 函数值 == 0)

    def constraint1(x):
        # 原题：x1^2 - x2 + x3^2 >= 0
        return x[0] ** 2 - x[1] + x[2] ** 2

    def constraint2(x):
        # 原题：x1 + x2^2 + x3^3 = 20 -> 变形为 x1 + x2^2 + x3^3 - 20 = 0
        return x[0] + x[1] ** 2 + x[2] ** 3 - 20

    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'eq', 'fun': constraint2}
    cons = [con1, con2]

    # --- 3. 定义变量边界 ---
    # (min, max)
    b = (0.0, None)  # 0 <= x
    bnds = (b, b, b)

    # --- 4. 设置初始猜测值 (Initial Guess) ---
    # 这一步非常重要，建议根据约束条件大概估一个
    x0 = [1, 1, 1]

    # --- 5. 求解 ---
    # SLSQP 是处理非线性约束最常用的算法
    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

    # --- 6. 结果输出 ---
    if res.success:
        print("优化成功！")
        print(f"最优解 x: {res.x}")
        print(f"最小目标函数值 f(x): {res.fun}")
    else:
        print("优化失败，原因:", res.message)


# ================= 使用示例 =================

if __name__ == "__main__":
    solve_nonlinear_programming()