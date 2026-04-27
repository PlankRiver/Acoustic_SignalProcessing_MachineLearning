import pandas as pd
import numpy as np


def entropy_weight_method(data, direction_list):
    """
    熵权法计算权重
    :param data: pandas DataFrame, 原始数据 (行=样本, 列=指标)
    :param direction_list: list, 指示每列是正向还是负向。
                           1 代表正向指标(越大越好), 0 代表负向指标(越小越好)
    :return: pandas Series, 各指标的权重
    """

    # 为了不破坏原数据，拷贝一份
    df = data.copy().astype(float)
    m_columns = df.columns
    n_rows = df.shape[0]

    # --- 1. 数据标准化 (Min-Max Scaling) ---
    # 这一步是为了消除量纲，并将数据映射到 [0, 1]
    for i, col in enumerate(m_columns):
        direction = direction_list[i]
        max_val = df[col].max()
        min_val = df[col].min()

        # 防止分母为0 (如果某列所有值都一样，max-min=0)
        if max_val == min_val:
            df[col] = 0.5  # 如果所有值一样，给予中间值，其实该指标权重最后会是0
        else:
            if direction == 1:  # 正向指标
                df[col] = (df[col] - min_val) / (max_val - min_val)
            elif direction == 0:  # 负向指标
                df[col] = (max_val - df[col]) / (max_val - min_val)
            else:
                raise ValueError(f"direction_list 中只能包含 0 或 1，位置 {i} 错误")

    # --- 2. 计算比重 P_ij ---
    # 有时候标准化后会出现0，导致后面 log(0) 报错。
    # 这里做一个平移，或者在计算log时加极小值。通常的做法是平移一丢丢，或者标准化范围设为 [0.001, 1]
    # 这里我们采用在 log 计算时加 epsilon 的方法，保持数据纯度

    # 计算每列的总和
    sum_cols = df.sum(axis=0)

    # 归一化得到 P_ij
    # 如果某列和为0（极端情况），则该列比重设为 1/n
    P = df / sum_cols
    P = P.fillna(1 / n_rows)  # 填充可能产生的NaN

    # --- 3. 计算熵值 E_j ---
    k = 1 / np.log(n_rows)
    # 加上 1e-6 避免 log(0) 导致 -inf
    E = -k * (P * np.log(P + 1e-6)).sum(axis=0)

    # --- 4. 计算差异系数 D_j ---
    D = 1 - E

    # --- 5. 计算权重 W_j ---
    # 如果所有指标的差异系数和为0（非常罕见），则平均分配权重
    if D.sum() == 0:
        W = pd.Series(np.ones(len(m_columns)) / len(m_columns), index=m_columns)
    else:
        W = D / D.sum()

    return W


# ================= 使用示例 =================

if __name__ == "__main__":
    # 1. 准备数据
    # 假设我们评价 3 个城市的 4 个指标
    # 指标: [GDP(正), 犯罪率(负), 绿化率(正), 交通拥堵指数(负)]
    data = {
        'GDP': [100, 120, 90],  # 越大越好
        '犯罪率': [5.2, 4.8, 6.0],  # 越小越好
        '绿化率': [30, 35, 25],  # 越大越好
        '拥堵指数': [8.5, 9.0, 7.5]  # 越小越好
    }
    df = pd.DataFrame(data, index=['城市A', '城市B', '城市C'])
    print("原始数据:\n", df)

    # 2. 定义指标方向 (1=正向, 0=负向)
    # 对应列顺序: GDP(1), 犯罪率(0), 绿化率(1), 拥堵指数(0)
    directions = [1, 0, 1, 0]

    # 3. 调用函数
    weights = entropy_weight_method(df, directions)

    # 4. 输出结果
    print("-" * 30)
    print("各指标权重 (熵权法):")
    print(weights)

    # (进阶) 计算各城市的综合得分
    # 简单的加权求和: Score = Norm_Data * Weights
    # 注意：这里需要用标准化后的数据乘权重，为了演示简单，我们重新简单标准化一下
    # 实际操作中，可以将 entropy_weight_method 函数修改为同时返回标准化数据

    print("-" * 30)
    print("权重最高的指标是:", weights.idxmax())