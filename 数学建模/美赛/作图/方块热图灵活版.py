import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

# 1. 准备模拟数据 (10x10 矩阵)
np.random.seed(42)
data = np.random.rand(10, 10)
# 模拟一些空值或极小值
data[data < 0.1] = 0

# 2. 定义颜色映射 (还原原图的离散色阶)
# 颜色从浅蓝到深红
colors = [
    '#ffffff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
    '#2171b5', '#084594', '#006d2c', '#2ca25f', '#74c476', '#a1d99b',
    '#e5f5e0', '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c',
    '#fc4e2a', '#e31a1c', '#bd0026'
]
# 这里为了简化，我们使用一个预设的渐变色，你可以根据需要调整
cmap = plt.get_cmap('jet', 20)

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 设置坐标轴范围
N = data.shape[0]
ax.set_xlim(0.5, N + 0.5)
ax.set_ylim(0.5, N + 0.5)

# 4. 核心逻辑：循环绘制每一个方块
for i in range(N):
    for j in range(N):
        val = data[i, j]
        if val > 0:
            # 计算方块大小：数值越大方块越大 (最大为 0.9，留出一点缝隙)
            size = val * 0.9
            # 计算起始位置，使方块居中
            x = (j + 1) - size / 2
            y = (i + 1) - size / 2

            # 绘制方块
            rect = Rectangle((x, y), size, size,
                             facecolor=cmap(val),
                             edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

# 5. 绘制黑色的网格线 (模仿原图风格)
for k in range(N + 1):
    ax.axhline(k + 0.5, color='black', linewidth=1)
    ax.axvline(k + 0.5, color='black', linewidth=1)

# 6. 图表美化
ax.set_title('Square Heatmap Plot', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('K (w)', fontsize=14)
ax.set_ylabel('Samples', fontsize=14)

# 设置刻度标签居中
ax.set_xticks(np.arange(1, N + 1))
ax.set_yticks(np.arange(1, N + 1))

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks(np.arange(0, 1.1, 0.1))

# 保持纵横比为正方形
ax.set_aspect('equal')

# 7. 布局与保存
plt.tight_layout()
# plt.savefig('square_heatmap.pdf', bbox_inches='tight')
plt.show()