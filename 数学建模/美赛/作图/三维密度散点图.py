import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置全局字体（美赛学术风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据 (模拟一个类似“墨西哥帽”的分布)
# 生成随机采样点，使图表看起来像散点而非平滑曲面
np.random.seed(42)
n_points = 10000
x = np.random.uniform(-10, 10, n_points)
y = np.random.uniform(-10, 10, n_points)
r = np.sqrt(x**2 + y**2)
# 模拟 Z 值，使用 sinc 函数逻辑
z = np.sin(r) / (r + 0.1)

# 3. 创建 3D 画布
fig = plt.figure(figsize=(12, 9), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 4. 绘制三维散点
# c: 颜色映射到 Z 值
# s: 点的大小，点多时调小一点（如 2-5）
# cmap: 使用 'jet' 还原原图颜色，或者 'RdYlBu_r' 更具现代学术感
scatter = ax.scatter(x, y, z, c=z, s=4, cmap='jet', alpha=0.8)

# 5. 美化细节
ax.set_title('Density Scatter3', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('XAxis', fontsize=12)
ax.set_ylabel('YAxis', fontsize=12)
ax.set_zlabel('ZAxis', fontsize=12)

# 设置轴范围
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-0.4, 1)

# 设置网格背景为白色（还原清爽风格）
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# 调整视角 (仰角 elev, 方位角 azim)
ax.view_init(elev=25, azim=-45)

# 6. 添加颜色条 (Colorbar)
# 模拟原图那种带刻度和颜色分级的 Colorbar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=15, pad=0.1)
# 设置颜色条标签
# 如果你想让颜色条显示 10-100 的刻度，可以对映射数值做线性变换
cbar.ax.set_ylabel('Intensity / Value', rotation=270, labelpad=15)

# 7. 调整布局并显示
plt.tight_layout()
# plt.savefig('3d_density_scatter.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()