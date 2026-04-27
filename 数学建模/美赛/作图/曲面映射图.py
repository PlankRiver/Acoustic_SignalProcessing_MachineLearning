import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置全局字体（美赛建议使用 Serif 字体，如 Times New Roman）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据 (模拟一个复杂的地形函数)
x = np.linspace(10, 20, 100)
y = np.linspace(15, 25, 100)
X, Y = np.meshgrid(x, y)

# 模拟 Z 值 (这里使用正弦函数组合出多个峰值)
Z = 35 + 8 * np.sin(0.5 * X) * np.cos(0.5 * Y) + 5 * np.sin(0.2 * X * Y / 10)

# 3. 创建画布
fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 4. 绘制 3D 曲面
# cmap: 选用 'RdBu_r' (红-白-蓝) 还原原图配色
# linewidth: 设置网格线宽度，antialiased使边缘平滑
surf = ax.plot_surface(X, Y, Z, cmap='RdBu_r', edgecolor='none',
                       linewidth=0, antialiased=True, alpha=0.8, zorder=1)

# 5. 绘制底部投影 (Contourf 填充等高线图)
# zdir='z' 表示向 Z 方向投影，offset 设置投影所在的平面高度
cset = ax.contourf(X, Y, Z, zdir='z', offset=20, cmap='RdBu_r', alpha=0.5)

# 6. 美化坐标轴
ax.set_zlim(20, 55) # 设置 Z 轴范围，确保投影在底部留有空间
ax.set_xlabel('X Axis (X轴)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Y Axis (Y轴)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_zlabel('Z Axis (Z轴)', fontsize=12, fontweight='bold', labelpad=10)

# 设置网格背景颜色为白色 (还原原图清爽质感)
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# 7. 调整视角 (elev是俯仰角，azim是旋转角)
ax.view_init(elev=25, azim=45)

# 8. 添加颜色条 (Colorbar)
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Z Value (Z值)', fontsize=10)

# 9. 调整布局并显示
plt.tight_layout()
# plt.savefig('3d_surface_projection.pdf', bbox_inches='tight') # 建议存为PDF用于论文
plt.show()