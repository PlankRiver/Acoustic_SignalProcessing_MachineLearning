import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt


def rsr_method(data, direction_list, plot=False):
    """
    秩和比综合评价法 (RSR)
    :param data: pandas DataFrame, 原始数据 (n行样本 x m列指标)
    :param direction_list: list, 指标方向 (1:正向/越大越好, 0:负向/越小越好)
    :param plot: bool, 是否绘制回归拟合图
    :return: pandas DataFrame (包含RSR值、拟合值、分档等级)
    """
    df = data.copy().astype(float)
    n, m = df.shape

    # --- 1. 编秩 (Rank) ---
    # 也就是把原始数据变成排名
    rank_df = pd.DataFrame(index=df.index, columns=df.columns)

    for i, col in enumerate(df.columns):
        direction = direction_list[i]
        if direction == 1:  # 正向: 值越大，排名越大(越好) -> method='average'处理并列
            rank_df[col] = df[col].rank(method='average', ascending=True)
        else:  # 负向: 值越小，排名越大(越好)
            rank_df[col] = df[col].rank(method='average', ascending=False)

    # --- 2. 计算 RSR 值 ---
    # RSR = Sum(Ranks) / (m * n)
    df['Sum_Rank'] = rank_df.sum(axis=1)
    df['RSR'] = df['Sum_Rank'] / (m * n)

    # --- 3. 计算 Probit (概率单位) ---
    # 为了做回归，需要对 RSR 进行排序，并计算累积频率

    # 这里的排序是为了计算频率分布，不是最终排名
    # 将 RSR 提取出来单独处理
    rsr_series = df['RSR'].sort_values()

    # 计算累计频率 p = (Rank_of_RSR - 0.5) / n
    # 注意：这里的 Rank_of_RSR 是 RSR 值本身的排名 (1, 2, ..., n)
    rsr_ranks = np.arange(1, n + 1)
    proportions = (rsr_ranks - 0.5) / n

    # 计算 Probit = NormInv(p) + 5
    probits = norm.ppf(proportions) + 5

    # 将计算结果匹配回原 DataFrame (有点绕，因为要对应索引)
    # 创建一个临时 DF 来存 Probit
    temp_df = pd.DataFrame({
        'RSR': rsr_series.values,
        'Probit': probits
    }, index=rsr_series.index)

    # 合并回去
    df = df.join(temp_df[['Probit']])

    # --- 4. 线性回归拟合 ---
    # 模型: RSR = a + b * Probit
    # 这里的自变量是 Probit (X), 因变量是 RSR (Y)
    slope, intercept, r_value, p_value, std_err = linregress(df['Probit'], df['RSR'])

    print(f"回归方程: RSR = {intercept:.4f} + {slope:.4f} * Probit")
    print(f"拟合优度 R^2: {r_value ** 2:.4f}")

    # 计算拟合值 (Fitted RSR)
    df['Fitted_RSR'] = intercept + slope * df['Probit']

    # --- 5. 分档排序 (Leveling) ---
    # 常用分档标准 (根据 Probit 值划分):
    # Probit < 4: 差
    # 4 <= Probit < 6: 中
    # 6 <= Probit < 8: 良
    # Probit >= 8: 优 (一般很少有>=8，除非样本量极大)

    def get_level(probit):
        if probit < 4:
            return '差 (Level 1)'
        elif probit < 6:
            return '中 (Level 2)'
        elif probit < 8:
            return '良 (Level 3)'
        else:
            return '优 (Level 4)'

    df['Level'] = df['Probit'].apply(get_level)
    df['Rank'] = df['Fitted_RSR'].rank(ascending=False)  # 按拟合值排名

    # --- 6. 绘图 (可选) ---
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['Probit'], df['RSR'], label='Original RSR')
        plt.plot(df['Probit'], df['Fitted_RSR'], color='red', label='Regression Line')
        plt.xlabel('Probit')
        plt.ylabel('RSR')
        plt.title('RSR Distribution Regression')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df[['RSR', 'Probit', 'Fitted_RSR', 'Level', 'Rank']].sort_values(by='Rank')


# ================= 使用示例 =================

if __name__ == "__main__":
    # 场景：评价 7 个科室的医疗质量
    # 指标：治愈率(正), 死亡率(负), 平均住院日(负), 病床周转次(正)
    data = {
        '治愈率(%)': [85, 90, 88, 82, 95, 80, 89],
        '死亡率(%)': [2.1, 1.5, 1.8, 2.5, 1.0, 3.0, 1.6],
        '平均住院日': [12, 10, 11, 14, 9, 15, 10.5],
        '病床周转次': [15, 18, 16, 12, 20, 10, 17]
    }
    df = pd.DataFrame(data, index=['科室A', '科室B', '科室C', '科室D', '科室E', '科室F', '科室G'])

    print("原始数据:\n", df)

    # 定义方向: 治愈率(1), 死亡率(0), 住院日(0), 周转次(1)
    directions = [1, 0, 0, 1]

    # 运行 RSR 模型
    results = rsr_method(df, directions, plot=True)

    print("-" * 30)
    print("RSR 评价结果:")
    print(results)