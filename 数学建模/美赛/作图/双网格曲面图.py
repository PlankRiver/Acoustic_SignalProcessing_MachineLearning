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

# 上层网格曲面数据（红-橙-蓝渐变）
Z1 = 550 + 100 * np.exp(-((X/800)**2 + (Y/800)**2)) * np.cos(X/500) * np.sin(Y/500)
# 下层网格曲面数据（绿-蓝渐变）
Z2 = 100 * np.exp(-((X/1000)**2 + (Y/1000)**2)) * np.cos(X/600) * np.sin(Y/600)

# ----------------------
# 2. 绘图设置
# ----------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# ----------------------
# 3. 绘制双层网格曲面（修正核心：分开绘制填充曲面+网格线，解决报错）
# ----------------------
# 上层：先画填充曲面（带颜色渐变），再叠加透明网格线
surf1_fill = ax.plot_surface(X, Y, Z1, rstride=2, cstride=2, cmap=cm.coolwarm,
                             alpha=0.6, linewidth=0, antialiased=True)
surf1_wire = ax.plot_wireframe(X, Y, Z1, rstride=2, cstride=2, color='black',
                               alpha=0.3, linewidth=0.5)

# 下层：先画填充曲面（带颜色渐变），再叠加透明网格线
surf2_fill = ax.plot_surface(X, Y, Z2, rstride=2, cstride=2, cmap=cm.viridis,
                             alpha=0.6, linewidth=0, antialiased=True)
surf2_wire = ax.plot_wireframe(X, Y, Z2, rstride=2, cstride=2, color='black',
                               alpha=0.3, linewidth=0.5)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
ax.set_title('DoubleMesh Plot', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')

# 添加两个颜色条
cbar1 = fig.colorbar(surf1_fill, ax=ax, shrink=0.5, aspect=10, pad=0.05)
cbar1.set_label('Upper Mesh Value', fontsize=10, fontweight='bold')
cbar1.set_ticks(np.linspace(Z1.min(), Z1.max(), 5))

cbar2 = fig.colorbar(surf2_fill, ax=ax, shrink=0.5, aspect=10, pad=0.15)
cbar2.set_label('Lower Mesh Value', fontsize=10, fontweight='bold')
cbar2.set_ticks(np.linspace(Z2.min(), Z2.max(), 5))

# 设置视角
ax.view_init(elev=30, azim=-45)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()