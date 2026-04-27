import pandas as pd
import numpy as np


def topsis(data, direction_list, weights=None):
    """
    TOPSIS 综合评价模型 (支持自动计算熵权)
    :param data: pandas DataFrame, 原始数据
    :param direction_list: list, 指标方向 (1:正向/越大越好, 0:负向/越小越好)
    :param weights: list or np.array, 权重列表。如果为 None，则自动使用熵权法计算。
    :return: pandas DataFrame (包含得分和排名)
    """
    df = data.copy().astype(float)
    n, m = df.shape

    # --- 1. 指标正向化处理 ---
    # 统一转化为“越大越好”
    for i, col in enumerate(df.columns):
        direction = direction_list[i]
        if direction == 0:  # 负向指标 -> 正向转化
            # 使用 max - x 的方式转化
            df[col] = df[col].max() - df[col]
        # 如果是中间型或区间型，建议在传入函数前在 Excel 或外部处理好，保持代码简洁

    # --- 2. 数据标准化 (向量归一化) ---
    # TOPSIS 标准做法: x / sqrt(sum(x^2))
    norm_df = df / np.sqrt((df ** 2).sum())

    # --- 3. 确定权重 ---
    if weights is None:
        print("提示: 未传入权重，将自动使用【熵权法】计算客观权重...")
        # 临时使用 Min-Max 归一化来算熵权 (熵权法对负数敏感，向量归一化可能有问题，这里局部处理)
        # 注意：为了代码简便，这里直接用向量归一化的数据算熵权虽不严谨但可用，
        # 更严谨的做法是重新做一次 Min-Max 归一化算熵权。这里简化演示：

        # 简单处理：将 norm_df 归一化到 [0, 1] 用于算熵 (避免 log 负数或 0)
        temp_norm = (df - df.min()) / (df.max() - df.min())
        # 计算比重
        P = temp_norm / temp_norm.sum()
        P = P.fillna(0)  # 处理分母为0
        # 计算熵
        k = 1 / np.log(n)
        E = -k * (P * np.log(P + 1e-6)).sum()
        # 差异系数
        D = 1 - E
        calc_weights = D / D.sum()
        weights = calc_weights.values
        print("自动计算的权重:", np.round(weights, 4))
    else:
        weights = np.array(weights)
        if len(weights) != m:
            raise ValueError("权重长度与指标列数不一致")

    # --- 4. 确定正负理想解 ---
    # 因为已经正向化了，所以正理想解就是 max，负理想解就是 min
    # 每一列的最大值
    Z_plus = norm_df.max()
    # 每一列的最小值
    Z_minus = norm_df.min()

    # --- 5. 计算距离 ---
    # 权重加权距离
    # 技巧: (x - max)^2 * w  <-- 这种写法是错的，应该是 sum( w * (x-max)^2 )
    # 或者先加权: weighted_data = data * w

    weighted_norm_df = norm_df * weights
    weighted_Z_plus = Z_plus * weights
    weighted_Z_minus = Z_minus * weights

    # 计算 D+ (与最优解的欧氏距离)
    # axis=1 表示按行求和
    D_plus = np.sqrt(((weighted_norm_df - weighted_Z_plus) ** 2).sum(axis=1))

    # 计算 D- (与最劣解的欧氏距离)
    D_minus = np.sqrt(((weighted_norm_df - weighted_Z_minus) ** 2).sum(axis=1))

    # --- 6. 计算得分 ---
    # S = D- / (D+ + D-)
    score = D_minus / (D_plus + D_minus)

    # 整理输出
    result = pd.DataFrame({
        'D+': D_plus,
        'D-': D_minus,
        '得分': score,
        '排名': score.rank(ascending=False)
    })

    return result


# ================= 使用示例 =================

if __name__ == "__main__":
    # 1. 准备数据
    # 案例：评价 4 款水质监测点的水质
    # 指标：含氧量(越高越好), PH值差异(越低越好), 细菌总数(越低越好), 矿物质(越高越好)
    data = {
        '含氧量': [7.2, 6.5, 8.1, 5.5],  # 正向 (1)
        'PH差值': [0.1, 0.5, 0.2, 0.8],  # 负向 (0) - 假设原始PH是7.0为最好，这里给出的是|PH-7|
        '细菌数': [120, 200, 50, 300],  # 负向 (0)
        '矿物质': [15, 12, 18, 10]  # 正向 (1)
    }
    df = pd.DataFrame(data, index=['A点', 'B点', 'C点', 'D点'])
    print("原始数据:\n", df)

    # 2. 定义方向
    directions = [1, 0, 0, 1]

    # 3. 运行模型 (不传 weights，让它自动用熵权法)
    print("-" * 30)
    print("模式1: 自动计算熵权 (Entropy-TOPSIS)")
    results_auto = topsis(df, directions, weights=None)
    print(results_auto.sort_values(by='排名'))

    # 4. 运行模型 (手动指定权重)
    # 比如 AHP 算出来权重是 [0.2, 0.3, 0.4, 0.1]
    print("-" * 30)
    print("模式2: 手动指定权重 (AHP-TOPSIS)")
    manual_weights = [0.2, 0.3, 0.4, 0.1]
    results_manual = topsis(df, directions, weights=manual_weights)
    print(results_manual.sort_values(by='排名'))