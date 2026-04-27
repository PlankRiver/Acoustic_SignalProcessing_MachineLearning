import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# ----------------------
# 1. 生成模拟数据（替换成你的真实数据）
# ----------------------
# 创建X, Y网格
x = np.linspace(-2000, 2000, 100)
y = np.linspace(-2000, 2000, 100)
X, Y = np.meshgrid(x, y)

# 上层曲面数据（绿色渐变）
Z1 = 550 + 100 * np.exp(-((X/800)**2 + (Y/800)**2)) * np.cos(X/500) * np.sin(Y/500)
# 下层曲面数据（棕蓝渐变）
Z2 = 100 * np.exp(-((X/1000)**2 + (Y/1000)**2)) * np.cos(X/600) * np.sin(Y/600)

# ----------------------
# 2. 绘图设置
# ----------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# ----------------------
# 3. 绘制双层曲面
# ----------------------
# 上层曲面（绿色配色）
surf1 = ax.plot_surface(X, Y, Z1, cmap=cm.YlGn, edgecolor='none', alpha=0.9)
# 下层曲面（棕蓝配色）
surf2 = ax.plot_surface(X, Y, Z2, cmap=cm.BrBG, edgecolor='none', alpha=0.9)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加标题
ax.set_title('DoubleSurface Plot', fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴标签
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')

# 添加颜色条
cbar1 = fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, pad=0.05)
cbar1.set_label('Upper Surface Value', fontsize=10, fontweight='bold')
cbar2 = fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, pad=0.15)
cbar2.set_label('Lower Surface Value', fontsize=10, fontweight='bold')

# 设置视角
ax.view_init(elev=30, azim=-45)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()