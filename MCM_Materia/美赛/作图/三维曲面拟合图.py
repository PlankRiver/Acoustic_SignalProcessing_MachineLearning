import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置全局字体（美赛建议使用 Serif 字体，如 Times New Roman）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 准备数据 (模拟一个拟合曲面)
# X轴范围约 0-50, Y轴范围约 20-45
x = np.linspace(0, 50, 100)
y = np.linspace(20, 45, 100)
X, Y = np.meshgrid(x, y)

# 模拟 Z 值：一个随 X 指数上升并随 Y 波动的函数
# 为了还原原图范围 (-50 到 200)，做一些数值偏移
Z = 150 * np.exp(-X/15) + 30 * np.sin(Y/5) + 20

# 3. 创建画布
fig = plt.figure(figsize=(12, 9), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 4. 绘制 3D 曲面
# cmap: 使用 'viridis' 或 'parula' (parula需额外安装)，这里用内置的 'viridis'
# antialiased: 使表面更平滑
# edgecolor: 设置网格线颜色，这里用黑色细线增强立体感
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                       linewidth=0.5, edgecolor='black',
                       alpha=0.9, antialiased=True)

# 5. 美化细节与坐标轴
ax.set_title('3D Numerical Fitting Curve (三维数值拟合曲线)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X Axis (X轴)', fontsize=12)
ax.set_ylabel('Y Axis (Y轴)', fontsize=12)
ax.set_zlabel('Z Axis (Z轴)', fontsize=12)

# 设置轴范围，还原原图特征
ax.set_xlim(0, 50)
ax.set_ylim(20, 45)
ax.set_zlim(-50, 20) # 这里的刻度根据你的数据调整

# 调整视角 (仰角 elev, 方位角 azim)
ax.view_init(elev=25, azim=45)

# 设置背景颜色为白色 (还原 MATLAB 清爽质感)
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')
# 开启网格线
ax.grid(True, linestyle='--', alpha=0.3)

# 6. 添加颜色条 (Colorbar)
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.1)
cbar.set_label('z value (z值)', fontsize=11)

# 7. 自动布局并显示
plt.tight_layout()
# plt.savefig('fitting_surface.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()