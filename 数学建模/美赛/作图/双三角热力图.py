import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ----------------------
# 1. 生成模拟数据（替换成你的真实相关系数矩阵）
# ----------------------
# 特征名称（和你的图一致）
labels = ['Carb', 'Wt', 'Hp', 'Cyl', 'Disp', 'Qsec', 'Vs', 'Mpg', 'Drat', 'Gear']
n = len(labels)

# 生成随机相关系数矩阵（对称矩阵）
np.random.seed(42)
corr = np.random.rand(n, n)
corr = (corr + corr.T) / 2  # 保证对称
np.fill_diagonal(corr, 1)   # 对角线为1

# ----------------------
# 2. 创建双三角遮罩
# ----------------------
mask_upper = np.triu(np.ones_like(corr, dtype=bool))
mask_lower = np.tril(np.ones_like(corr, dtype=bool))

# ----------------------
# 3. 定义双配色方案（和你的图一致）
# ----------------------
# 上三角配色：从白到绿再到紫
colors_upper = ['#FFFFFF', '#F9F9C5', '#C2EABD', '#76B852', '#3A7D44', '#4A235A', '#884EA0']
cmap_upper = LinearSegmentedColormap.from_list('custom_upper', colors_upper, N=256)

# 下三角配色：从白到红再到紫
colors_lower = ['#FFFFFF', '#FADBD8', '#F1948A', '#E74C3C', '#C0392B', '#884EA0', '#4A235A']
cmap_lower = LinearSegmentedColormap.from_list('custom_lower', colors_lower, N=256)

# ----------------------
# 4. 绘制双三角热图
# ----------------------
plt.figure(figsize=(12, 10))

# 绘制上三角
sns.heatmap(corr, mask=mask_lower, cmap=cmap_upper, annot=False, fmt='.2f',
            linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Upper Triangle Value'})

# 绘制下三角
sns.heatmap(corr, mask=mask_upper, cmap=cmap_lower, annot=False, fmt='.2f',
            linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Lower Triangle Value'})

# ----------------------
# 5. 美化图表（美赛必备）
# ----------------------
plt.title('Double Triangle Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xticks(np.arange(n) + 0.5, labels, rotation=45, ha='right', fontsize=11)
plt.yticks(np.arange(n) + 0.5, labels, rotation=0, fontsize=11)

# 优化布局
plt.tight_layout()

# ----------------------
# 6. 显示图表
# ----------------------
plt.show()