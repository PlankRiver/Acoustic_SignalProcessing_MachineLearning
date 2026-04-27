import pandas as pd
import numpy as np


def grey_relational_analysis(data, directions=None, rho=0.5):
    """
    灰色关联分析法 (用于评价排名)
    :param data: pandas DataFrame, 行=评价对象, 列=指标
    :param directions: list, 指标方向 (1=正向/越大越好, 0=负向/越小越好)。
                       如果不填，默认全为正向。
    :param rho: 分辨系数，通常取 0.5
    :return: pandas DataFrame, 包含各指标的关联系数、关联度得分、排名
    """
    df = data.copy().astype(float)
    rows, cols = df.shape

    # 1. 默认方向处理
    if directions is None:
        directions = [1] * cols
    if len(directions) != cols:
        raise ValueError("directions 列表长度必须与列数一致")

    # 2. 数据标准化 (Min-Max Normalization)
    # 评价问题中，我们希望构建一个“全1”的参考序列，所以要把数据归一化到 [0, 1]
    df_norm = df.copy()
    for i, col in enumerate(df.columns):
        direction = directions[i]
        min_val = df[col].min()
        max_val = df[col].max()

        if max_val == min_val:
            df_norm[col] = 1.0  # 没区别就全是1
        else:
            if direction == 1:  # 正向
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:  # 负向
                df_norm[col] = (max_val - df[col]) / (max_val - min_val)

    # 3. 确定参考序列 (Reference Sequence)
    # 在标准化后，最优的参考值显然是每一列都是 1
    # 当然，也可以手动指定，但为了通用，我们默认最优就是1
    reference_seq = np.ones(cols)

    # 4. 计算绝对差值矩阵 (Delta)
    # axis=1 广播，每一行都减去参考序列
    delta = np.abs(df_norm - reference_seq)

    # 5. 获取两级最小差和最大差
    min_min = delta.min().min()
    max_max = delta.max().max()

    # 6. 计算关联系数矩阵 (Correlation Coefficients)
    # 公式: (min + rho * max) / (delta + rho * max)
    coeffs = (min_min + rho * max_max) / (delta + rho * max_max)

    # 7. 计算关联度 (Correlation Degree) -> 即最终得分
    # 求每一行的平均值
    scores = coeffs.mean(axis=1)

    # 8. 整理结果
    result = pd.DataFrame({
        '关联度(得分)': scores,
        '排名': scores.rank(ascending=False)
    })

    return result, coeffs


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：评价 5 个品牌的手机，4 个指标
    # 指标：价格(低好), 性能(高好), 颜值(高好), 重量(低好)
    data = {
        '价格': [5000, 4000, 3000, 8000, 2500],
        '性能跑分': [90, 85, 70, 98, 60],
        '颜值评分': [8.5, 9.0, 7.5, 9.5, 8.0],
        '重量(g)': [200, 180, 190, 220, 170]
    }
    df = pd.DataFrame(data, index=['华为', '小米', 'OPPO', '苹果', 'Vivo'])

    print("原始数据:\n", df)

    # 定义方向: 价格(0), 性能(1), 颜值(1), 重量(0)
    directions = [0, 1, 1, 0]

    # 运行算法
    results, detail_coeffs = grey_relational_analysis(df, directions, rho=0.5)

    print("-" * 30)
    print("各指标的关联系数 (Detail):\n", detail_coeffs)

    print("-" * 30)
    print("最终评价结果:\n", results.sort_values(by='排名'))