import numpy as np
from scipy.optimize import linprog


def branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, integer_vars):  #整数规划函数
    """
    分支定界法求解整数规划
    integer_vars: 需要为整数的变量索引列表
    """
    best_solution = None
    best_value = -np.inf
    queue = [(bounds, None)]  # (当前边界, 父节点)
    while queue:
        current_bounds, parent = queue.pop(0)
        # 求解线性松弛
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=current_bounds, method='highs')
        if not res.success:
            continue
        # 如果当前解不如已知最优解，剪枝
        if -res.fun <= best_value:
            continue
        # 检查整数约束
        all_integer = True
        fractional_var = None
        for i in integer_vars:
            if not np.isclose(res.x[i], round(res.x[i]), atol=1e-6):
                all_integer = False
                fractional_var = i
                break
        if all_integer:
            # 找到整数解
            if -res.fun > best_value:
                best_value = -res.fun
                best_solution = res.x
            continue
        # 分支：选择分数变量进行分支
        if fractional_var is not None:
            floor_val = np.floor(res.x[fractional_var])
            ceil_val = np.ceil(res.x[fractional_var])
            # 创建两个子问题
            bounds1 = current_bounds.copy()
            bounds1[fractional_var] = (current_bounds[fractional_var][0], floor_val)
            bounds2 = current_bounds.copy()
            bounds2[fractional_var] = (ceil_val, current_bounds[fractional_var][1])
            queue.append((bounds1, res.x))
            queue.append((bounds2, res.x))
    return best_solution, best_value


# 示例使用
c = [1,1,1,1,1,1]  # 最大化目标函数
A_ub = [[-1,0,0,0,0,-1],
        [-1,-1,0,0,0,0],
        [0,-1,-1,0,0,0],
        [0,0,-1,-1,0,0],
        [0,0,0,-1,-1,0],
        [0,0,0,0,-1,-1]]
b_ub = [-35,-40,-50,-45,-55,-30]
bounds = [(0, None)]*6
integer_vars = [i for i in range(6)]  # 所有变量都需要整数

solution, value = branch_and_bound(c, A_ub, b_ub, None, None, bounds, integer_vars)
value = int(value)
solution = [int(i) for i in solution]
print("分支定界法结果:")
print(f"最优值: {-value}")
print(f"最优解: {solution}")