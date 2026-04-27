import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# ----------------------
# 1. 生成模拟数据（替换成你的真实数据）
# ----------------------
# 样本和K轴数量
n_samples = 10
n_k = 10

# 生成随机相关系数矩阵（范围-1到1）
np.random.seed(42)
data = np.random.uniform(-1, 1, (n_samples, n_k))

# 样本名称（倒序，和图一致）
samples = [f'{i}' for i in range(n_samples, 0, -1)]
k_values = [f'{i}' for i in range(1, n_k+1)]

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(12, 12))

# 使用与原图一致的发散配色（从-1的青绿色到1的黄色）
cmap = cm.get_cmap('PiYG')

# ----------------------
# 3. 绘制方块热图
# ----------------------
im = ax.pcolormesh(
    data, cmap=cmap, edgecolors='black', linewidth=1,
    vmin=-1, vmax=1, shading='flat'
)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 设置坐标轴
ax.set_xticks(np.arange(n_k) + 0.5)
ax.set_yticks(np.arange(n_samples) + 0.5)
ax.set_xticklabels(k_values, fontsize=11, fontweight='bold')
ax.set_yticklabels(samples, fontsize=11, fontweight='bold')
ax.set_xlabel('K (w)', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Samples', fontsize=14, fontweight='bold', labelpad=15)

# 添加标题
ax.set_title('Correlation Square Heatmap Plot', fontsize=16, fontweight='bold', pad=20)

# 添加颜色条
cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.05)
cbar.set_label('Correlation Value', fontsize=12, fontweight='bold')
cbar.set_ticks(np.arange(-1, 1.1, 0.2))

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()