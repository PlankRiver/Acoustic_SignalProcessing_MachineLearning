import pulp


def solve_integer_programming():
    """
    0-1 规划与整数规划求解模板 (使用 PuLP)

    【场景模拟】：项目投资选择
    有 5 个项目 (A, B, C, D, E)，每个项目有投入成本和预期收益。
    资金有限，且有逻辑约束。

    数据：
    项目:   A    B    C    D    E
    成本:   10   20   30   15   25  (万元)
    收益:   15   35   50   20   40  (万元)

    约束：
    1. 总资金不超过 60 万元。
    2. A 和 B 是互斥的 (技术冲突，只能二选一，或都不选)。
    3. 选了 C 就必须选 D (D是C的配套设施)。
    4. E 项目如果选，必须建立在 A 项目已选的基础上 (A是E的前置)。
    """

    # 1. 定义问题
    # pulp.LpMaximize (求最大值) 或 pulp.LpMinimize (求最小值)
    prob = pulp.LpProblem("Project_Selection", pulp.LpMaximize)

    # 2. 定义变量
    # cat='Binary' 表示 0-1 变量
    # cat='Integer' 表示 整数变量
    # cat='Continuous' 表示 连续变量
    projects = ['A', 'B', 'C', 'D', 'E']
    costs = {'A': 10, 'B': 20, 'C': 30, 'D': 15, 'E': 25}
    profits = {'A': 15, 'B': 35, 'C': 50, 'D': 20, 'E': 40}

    # 创建一个字典，包含所有决策变量 x_A, x_B ...
    x = pulp.LpVariable.dicts("Select", projects, cat='Binary')

    # 3. 定义目标函数 (直接加到 prob 中)
    # 目标：最大化总收益
    prob += pulp.lpSum([profits[i] * x[i] for i in projects]), "Total_Profit"

    # 4. 定义约束条件

    # (1) 资金限制
    prob += pulp.lpSum([costs[i] * x[i] for i in projects]) <= 60, "Budget_Limit"

    # (2) A 和 B 互斥 (A + B <= 1)
    prob += x['A'] + x['B'] <= 1, "Mutually_Exclusive_A_B"

    # (3) 选 C 必选 D (C <= D  =>  若C=1则D必为1)
    prob += x['C'] <= x['D'], "C_requires_D"

    # (4) 选 E 必须以 A 为前提 (E <= A => 若E=1则A必为1; 若A=0则E必为0)
    prob += x['E'] <= x['A'], "E_requires_A"

    # 5. 求解
    # 使用默认求解器 (CBC)
    # msg=1 显示求解过程, msg=0 不显示
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 6. 输出结果
    print("-" * 30)
    print("求解状态:", pulp.LpStatus[prob.status])

    if pulp.LpStatus[prob.status] == 'Optimal':
        print("最大收益:", pulp.value(prob.objective))
        print("被选中的项目:")
        for i in projects:
            if pulp.value(x[i]) == 1:
                print(f"  - 项目 {i} (成本:{costs[i]}, 收益:{profits[i]})")

        # 验证一下资金
        total_cost = sum([costs[i] for i in projects if pulp.value(x[i]) == 1])
        print(f"实际总花费: {total_cost} 万元")
    else:
        print("无可行解")


# ================= 使用示例 =================

if __name__ == "__main__":
    solve_integer_programming()