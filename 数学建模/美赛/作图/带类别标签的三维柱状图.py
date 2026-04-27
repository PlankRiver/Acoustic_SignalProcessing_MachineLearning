import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# 1. 设置全局样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据 (以10x10网格为例)
x_dim = 10
y_dim = 10
x = np.arange(1, x_dim + 1)
y = np.arange(1, y_dim + 1)
_x, _y = np.meshgrid(x, y)
x_flat, y_flat = _x.ravel(), _y.ravel()

# 设置柱子的高度 (Variable3) - 这里生成一个中心较高的分布模拟原图
dz = np.exp(-((x_flat-5)**2 + (y_flat-5)**2)/15) * 15 + np.random.rand(100) * 2
# 设置柱子的底部位置 (全部从0开始)
z_bottom = np.zeros_like(dz)

# 设置柱子的粗细 (dx, dy)
dx = dy = 0.6

# 3. 设置分类与配色 (模拟原图的5个分类)
# 假设我们根据某种逻辑将这100根柱子分为5类
categories = (x_flat % 5) + 1  # 只是示例逻辑
colors_map = {
    1: '#8dd3c7', # 浅绿
    2: '#ffffb3', # 浅黄
    3: '#bebada', # 浅紫
    4: '#fb8072', # 浅红
    5: '#80b1d3'  # 浅蓝
}
bar_colors = [colors_map[cat] for cat in categories]

# 4. 绘图
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 绘制3D柱状图
# shade=False 可以让颜色更纯净，edgecolor增加黑色边框
bars = ax.bar3d(x_flat - dx/2, y_flat - dy/2, z_bottom, dx, dy, dz,
                color=bar_colors, edgecolor='black', linewidth=0.5, shade=True)

# 5. 美化坐标轴与视角
ax.set_title('Bar3withLabels Plot', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Variable1', fontsize=12)
ax.set_ylabel('Variable2', fontsize=12)
ax.set_zlabel('Variable3', fontsize=12)

# 设置刻度范围
ax.set_zlim(0, 20)
ax.set_xticks(x)
ax.set_yticks(y)

# 调整观察视角 (elev是俯仰角，azim是方位角)
ax.view_init(elev=25, azim=45)

# 设置背景颜色和网格
ax.set_facecolor('white')
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 6. 添加图例 (3D图中bar3d不支持直接图例，需手动创建)
legend_elements = [Patch(facecolor=colors_map[i], edgecolor='black', label=str(i))
                   for i in range(1, 6)]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 0.8),
          frameon=True, edgecolor='black', title_fontsize=12)

# 7. 自动调整并显示
plt.tight_layout()
# plt.savefig('3d_bar_chart.pdf', bbox_inches='tight')
plt.show()