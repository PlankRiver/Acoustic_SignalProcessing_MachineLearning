import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 构造数据：线性 + 非线性单调 + 异常值
np.random.seed(42)
x = np.linspace(0, 10, 50)
y_linear = 2 * x + np.random.normal(0, 1, 50)
y_monotone = np.exp(x/2)
y_outlier = y_linear.copy()
y_outlier[0] = 500  # 添加一个剧烈的异常值

data = pd.DataFrame({
    'X': x,
    'Linear_Y': y_linear,
    'Monotone_Y': y_monotone,
    'Outlier_Y': y_outlier
})

# 1. 计算 Pearson
corr_p, p_p = pearsonr(data['X'], data['Linear_Y'])
# 2. 计算 Spearman
corr_s, p_s = spearmanr(data['X'], data['Monotone_Y'])

print(f"线性数据 - Pearson: {corr_p:.4f} (P={p_p:.4f})")
print(f"指数数据 - Pearson: {pearsonr(data['X'], data['Monotone_Y'])[0]:.4f}")
print(f"指数数据 - Spearman: {corr_s:.4f} (捕捉到了单调性)")
print("-" * 30)
print(f"含异常值 - Pearson: {pearsonr(data['X'], data['Outlier_Y'])[0]:.4f} (受影响严重)")
print(f"含异常值 - Spearman: {spearmanr(data['X'], data['Outlier_Y'])[0]:.4f} (依然稳健)")