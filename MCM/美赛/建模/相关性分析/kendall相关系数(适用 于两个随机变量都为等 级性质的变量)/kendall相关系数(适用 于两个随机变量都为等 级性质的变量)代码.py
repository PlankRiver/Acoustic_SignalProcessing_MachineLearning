import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# 1. 构造定序数据 (例如：两名评委对 10 个产品的排名)
judge_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
judge_B = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9] # 排名非常接近

# 2. 计算 Kendall Tau
corr_k, p_k = kendalltau(judge_A, judge_B)

# 3. 对比 Spearman
corr_s, p_s = spearmanr(judge_A, judge_B)

print("-" * 30)
print(f"Kendall's Tau: {corr_k:.4f} (P-value: {p_k:.4f})")
print(f"Spearman's Rho: {corr_s:.4f} (P-value: {p_s:.4f})")
print("-" * 30)

# 4. 模拟含有大量平局(Ties)的情况 (如满意度调查 1-5 分)
survey_1 = [5, 4, 4, 3, 2, 5, 1, 3, 4, 5]
survey_2 = [5, 3, 4, 2, 2, 4, 1, 4, 4, 5]
corr_tie, p_tie = kendalltau(survey_1, survey_2)
print(f"存在平局时的 Kendall 系数: {corr_tie:.4f}")