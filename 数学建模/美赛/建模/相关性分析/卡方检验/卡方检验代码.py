import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_test(data_matrix):
    """
    执行卡方独立性检验
    :param data_matrix: 观测频数矩阵 (列联表)
    """
    # 1. 执行检验
    # 返回值：chi2(统计量), p(P值), dof(自由度), expected(期望频数矩阵)
    chi2, p, dof, expected = chi2_contingency(data_matrix)

    print("-" * 30)
    print(f"卡方统计量 (Chi2): {chi2:.4f}")
    print(f"P 值 (P-value): {p:.4f}")
    print(f"自由度 (df): {dof}")

    # 2. 结论判断
    alpha = 0.05
    if p < alpha:
        print("结论: P < 0.05, 拒绝原假设，两个变量之间存在显著相关性。")
    else:
        print("结论: P >= 0.05, 接受原假设，两个变量之间独立。")

    return chi2, p


# --- 示例数据 ---
# 抽烟 vs 患肺病
#           患病   不患病
# 抽烟者    [50,    10]
# 不抽烟者  [15,    80]
obs = np.array([[50, 10], [15, 80]])

chi2_val, p_val = chi_square_test(obs)