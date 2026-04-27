import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# 1. 设置全局字体（美赛学术风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据
x = np.linspace(0, np.pi, 100) # X轴：弧度角
y = np.arange(1, 8)            # Y轴：7条折线
# 生成 Z 轴数据：每一条线的波峰高度不同
zs = [np.sin(x) * (i + 1.2) for i in y]

# 3. 设置配色 (还原原图从紫到红的色系)
# 依次为：深紫、橙黄、天蓝、青绿、深绿、淡蓝、玫红
colors = ['#4B0082', '#FFA500', '#00FFFF', '#20B2AA', '#008000', '#4169E1', '#C71585'][::-1]

# 4. 绘图准备
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 构造多边形顶点
verts = []
for z_val in zs:
    # 为了让面积图底部封闭在 Z=0，我们需要在数据点首尾各加一个 (x, 0)
    path = list(zip(x, z_val))
    path = [(x[0], 0)] + path + [(x[-1], 0)]
    verts.append(path)

# 5. 核心绘图：使用 PolyCollection 绘制 3D 填充区域
# zs=y 表示每个多边形在 Y 轴上的位置
poly = PolyCollection(verts, facecolors=colors, alpha=0.5, edgecolors=colors, linewidths=1)
ax.add_collection3d(poly, zs=y, zdir='y')

# 6. 美化细节
ax.set_title('3D Area Plot Template', fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴标签
ax.set_xlabel('Radian (弧度角)', fontsize=12)
ax.set_ylabel('Line Count (折线数量)', fontsize=12)
ax.set_zlabel('Value (值)', fontsize=12)

# 设置轴范围
ax.set_xlim(0, np.pi)
ax.set_ylim(0.5, 7.5)
ax.set_zlim(0, 8)

# 自定义 Y 轴刻度标签为 line1, line2...
ax.set_yticks(y)
ax.set_yticklabels([f'line{i}' for i in y])

# 调整视角 (仰角 elev, 方位角 azim)
ax.view_init(elev=25, azim=-45)

# 设置背景颜色和网格（还原类似 Matlab 的清爽质感）
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')
ax.grid(True, linestyle='-', alpha=0.3)

# 7. 自动布局并显示
plt.tight_layout()
# plt.savefig('3d_area_plot.pdf', bbox_inches='tight') # 建议保存为PDF用于论文
plt.show()