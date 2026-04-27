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
corr_data = np.random.uniform(-1, 1, (n_samples, n_k))
# 生成气泡大小数据（范围0到1）
size_data = np.random.uniform(0.3, 1, (n_samples, n_k)) * 1000

# 样本名称（倒序，和图一致）
samples = [f'{i}' for i in range(n_samples, 0, -1)]
k_values = [f'{i}' for i in range(1, n_k+1)]

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(14, 12))

# 使用与原图一致的发散配色（从-1的深红色到1的深绿色）
cmap = cm.get_cmap('RdYlGn_r')

# ----------------------
# 3. 绘制相关性气泡热图
# ----------------------
# 遍历每个单元格绘制气泡
for i in range(n_samples):
    for j in range(n_k):
        ax.scatter(
            j + 1, i + 1,  # 气泡位置
            s=size_data[i, j],  # 气泡大小
            c=corr_data[i, j],  # 气泡颜色
            cmap=cmap,
            vmin=-1, vmax=1,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.9
        )

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 设置坐标轴
ax.set_xticks(np.arange(1, n_k+1))
ax.set_yticks(np.arange(1, n_samples+1))
ax.set_xticklabels(k_values, fontsize=11, fontweight='bold')
ax.set_yticklabels(samples, fontsize=11, fontweight='bold')
ax.set_xlabel('K (w)', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Samples', fontsize=14, fontweight='bold', labelpad=15)

# 添加网格线
ax.grid(True, linestyle='-', color='black', linewidth=1, alpha=1)

# 添加标题
ax.set_title('Correlation bubble Heatmap Plot', fontsize=18, fontweight='bold', pad=20)

# 添加颜色条
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=25, pad=0.05)
cbar.set_label('Correlation Value', fontsize=12, fontweight='bold')
cbar.set_ticks(np.arange(-1, 1.1, 0.2))

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()