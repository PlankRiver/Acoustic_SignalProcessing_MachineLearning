import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局样式（推荐美赛使用的学术字体）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备模拟数据 (假设有500个点)
# 实际套用时，请用你的数据替换这里
np.random.seed(42)
n_points = 500
x = np.random.uniform(0, 500, n_points)
y = np.random.uniform(0, 1, n_points)

# 3. 创建画布
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 4. 绘制散点图
# c: 颜色（使用了原图类似的粉橙色 #fb8072）
# s: 点的大小
# edgecolor: 点的边框颜色
# linewidths: 边框粗细
# alpha: 透明度（如果点太密集，可以调低透明度看清重叠情况）
scatter = ax.scatter(x, y, c='#fb8072', s=100,
                     edgecolor='black', linewidths=0.8,
                     alpha=0.9, zorder=3)

# 5. 美化细节
# 设置网格线（淡灰色实线，置于底层）
ax.grid(True, linestyle='-', color='#e0e0e0', linewidth=1, zorder=0)

# 只保留左侧和底部的轴（L型坐标系）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 设置刻度范围和步长
ax.set_xlim(-20, 550)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks(np.arange(0, 501, 100))
ax.set_yticks(np.arange(0, 1.1, 0.2))

# 设置刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=14)

# 添加标题和轴标签（美赛论文必须包含）
# ax.set_title('Distribution of Random Samples', fontsize=18, fontweight='bold', pad=15)
# ax.set_xlabel('Index / Parameter X', fontsize=15)
# ax.set_ylabel('Normalized Value / Y', fontsize=15)

# 6. 布局优化并显示
plt.tight_layout()
# plt.savefig('scatter_plot.pdf', bbox_inches='tight') # 建议保存为PDF或高分辨率PNG
plt.show()