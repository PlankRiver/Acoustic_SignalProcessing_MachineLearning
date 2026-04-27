import numpy as np
from scipy.optimize import linprog


def solve_linear_programming():
    """
    线性规划求解模板
    题目示例：
    目标：Max Z = 4x1 + 3x2  (求最大利润)

    约束条件：
    1. 2x1 + x2 <= 10  (原料A限制)
    2. x1 + x2 <= 8    (原料B限制)
    3. x2 <= 7         (产能限制)
    4. x1, x2 >= 0     (非负限制)
    """

    # --- 1. 定义目标函数系数 c ---
    # 注意：linprog 默认求【最小值】(Min)。
    # 如果题目求【最大值】，必须对系数取反！
    # 原题 Max Z = 4x1 + 3x2  -->  Min -Z = -4x1 - 3x2
    c = [-4, -3]

    # --- 2. 定义不等式约束 (A_ub * x <= b_ub) ---
    # 将所有约束整理成 <= 的形式
    # 2x1 + 1x2 <= 10
    # 1x1 + 1x2 <= 8
    # 0x1 + 1x2 <= 7
    A_ub = np.array([
        [2, 1],
        [1, 1],
        [0, 1]
    ])
    b_ub = np.array([10, 8, 7])

    # --- 3. 定义等式约束 (A_eq * x = b_eq) ---
    # 如果没有等式约束，就留 None
    # 假设题目有个约束 x1 + x2 = 5，则：
    # A_eq = [[1, 1]]
    # b_eq = [5]
    A_eq = None
    b_eq = None

    # --- 4. 定义变量边界 (Bounds) ---
    # (min, max)，None代表无穷大
    # x1 >= 0 -> (0, None)
    # x2 >= 0 -> (0, None)
    x0_bounds = (0, None)
    x1_bounds = (0, None)

    # --- 5. 求解 ---
    # method='highs' 是目前 scipy 中性能最好的求解器
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[x0_bounds, x1_bounds], method='highs')

    # --- 6. 结果输出 ---
    if res.success:
        print("优化成功！")
        print(f"最优解 (x1, x2): {res.x}")
        # 记得把目标函数值取反回来（如果之前取反了的话）
        print(f"最大目标函数值 (Z): {-res.fun}")
    else:
        print("优化失败，原因:", res.message)


# ================= 使用示例 =================

if __name__ == "__main__":
    solve_linear_programming()