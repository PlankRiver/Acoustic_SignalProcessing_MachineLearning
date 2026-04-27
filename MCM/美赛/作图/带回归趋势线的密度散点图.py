import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress

# 1. 生成或加载你的数据 (这里用模拟数据代替)
# 假设 x 是观测值，y 是模型估计值
np.random.seed(42)
x = np.random.normal(80, 30, 2000)
y = x + np.random.normal(0, 10, 2000) # 加上一些噪声

# 过滤掉超出范围的数据（可选，根据你的坐标轴范围）
mask = (x > 10) & (x < 150) & (y > 0) & (y < 150)
x, y = x[mask], y[mask]

# 2. 计算点密度 (这是实现颜色深浅的关键)
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# 3. 按密度排序（让密度高的点显示在最上层，视觉效果更好）
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# 4. 计算回归线 (Linear Regression)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.array([min(x), max(x)])
line_y = slope * line_x + intercept

# 5. 开始绘图
fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

# 绘制密度散点图
# cmap 使用 'RdYlBu_r' (红-黄-蓝 翻转) 比较接近原图
sc = ax.scatter(x, y, c=z, s=15, cmap='RdYlBu_r', edgecolor=None)

# 绘制回归趋势线
ax.plot(line_x, line_y, color='black', linestyle=':', linewidth=3,
        label='Regression line', zorder=10)

# 6. 图表美化
ax.set_title('Satellite-derived bathymetry', fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel('ICESat-2 bathymetric points in depth (m)', fontsize=14)
ax.set_ylabel('Estimated depth (m)', fontsize=14)

# 设置坐标轴范围和刻度
ax.set_xlim(0, 160)
ax.set_ylim(0, 160)
ax.set_xticks(np.arange(0, 161, 40))
ax.set_yticks(np.arange(0, 161, 40))

# 添加网格线
ax.grid(True, linestyle='-', alpha=0.4, linewidth=1)

# 添加颜色条 (Colorbar)
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label('Point Density', rotation=270, labelpad=15)

# 添加图例
ax.legend(loc='upper left', frameon=True, edgecolor='black',
          fontsize=14, borderpad=0.5, handlelength=3)

# 设置坐标轴刻度粗细
ax.tick_params(direction='out', length=6, width=1.5, labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 7. 布局与保存
plt.tight_layout()
# plt.savefig('density_scatter_plot.pdf', bbox_inches='tight')
plt.show()

# 输出 R² 值（美赛论文里记得写在正文或图中）
print(f"R-squared: {r_value**2:.4f}")