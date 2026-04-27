import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def solve_dynamic_programming():
    """
    动态规划通用模板：资源分配问题

    【题目场景】
    你有 5 台机器 (Resource)，要分配给 3 个工厂 (Stages: A, B, C)。
    每个工厂获得不同数量的机器，产生的利润如下表：

    机器数 | 工厂A利润 | 工厂B利润 | 工厂C利润
    0      | 0         | 0         | 0
    1      | 3         | 5         | 4
    2      | 7         | 10        | 6
    3      | 9         | 11        | 11
    4      | 12        | 11        | 12
    5      | 13        | 11        | 12

    求：如何分配这 5 台机器，使得总利润最大？
    """

    # 1. 准备数据
    # profit_table[i][j] 表示：第 i 个工厂分配 j 台机器的利润
    # 行：机器数量 (0~5)，列：工厂 (A, B, C)
    profit_table = np.array([
        [0, 0, 0],  # 0台
        [3, 5, 4],  # 1台
        [7, 10, 6],  # 2台
        [9, 11, 11],  # 3台
        [12, 11, 12],  # 4台
        [13, 11, 12]  # 5台
    ])

    total_resources = 5  # 总资源数 (状态上限)
    num_stages = 3  # 阶段数 (工厂数)

    # 2. 初始化 DP 表
    # dp[k][w] 表示：只考虑前 k+1 个工厂，分配资源总量为 w 时的最大利润
    # 维度: (工厂数, 资源数+1)
    dp = np.zeros((num_stages, total_resources + 1))

    # record[k][w] 用于记录路径：达到 dp[k][w] 时，给第 k 个工厂分了多少台机器？
    record = np.zeros((num_stages, total_resources + 1), dtype=int)

    # 3. 处理第 0 个阶段 (边界条件)
    # 对于第一个工厂 A，给它多少资源，利润就是查表得多少，没有“前一个”
    for w in range(total_resources + 1):
        dp[0][w] = profit_table[w][0]
        record[0][w] = w  # 给多少就是多少

    # 4. 状态转移 (核心循环)
    # 从第 1 个工厂开始遍历到最后一个
    for k in range(1, num_stages):
        # 遍历当前总资源限制 w (从0到5)
        for w in range(total_resources + 1):

            max_val = -1
            best_allocation = 0

            # 决策变量 u: 给当前第 k 个工厂分配多少台？ (0 到 w 台)
            for u in range(w + 1):
                # 状态转移方程：
                # 当前总利润 = (第k个工厂分u台的利润) + (前k-1个工厂分 w-u 台的最大利润)
                current_profit = profit_table[u][k] + dp[k - 1][w - u]

                if current_profit > max_val:
                    max_val = current_profit
                    best_allocation = u

            dp[k][w] = max_val
            record[k][w] = best_allocation

    # 5. 结果回溯 (Backtracking) - 找出具体的分配方案
    allocation_plan = [0] * num_stages
    remaining_res = total_resources

    # 从最后一个阶段往前推
    for k in range(num_stages - 1, -1, -1):
        current_alloc = record[k][remaining_res]
        allocation_plan[k] = current_alloc
        remaining_res -= current_alloc

    # 6. 输出结果
    print("-" * 30)
    print(f"最大总利润: {dp[num_stages - 1][total_resources]}")
    print("最优分配方案:")
    for i in range(num_stages):
        factory_name = chr(65 + i)  # A, B, C
        print(
            f"  - 工厂 {factory_name} 分配: {allocation_plan[i]} 台机器 (利润: {profit_table[allocation_plan[i]][i]})")

    print("-" * 30)
    # 打印 DP 表格 (方便调试/放到论文里)
    df_dp = pd.DataFrame(dp, columns=[f"资源{i}" for i in range(total_resources + 1)],
                         index=[f"前{i + 1}个工厂" for i in range(num_stages)])
    print("过程表 (DP Table):")
    print(df_dp)


# ================= 使用示例 =================

if __name__ == "__main__":
    solve_dynamic_programming()