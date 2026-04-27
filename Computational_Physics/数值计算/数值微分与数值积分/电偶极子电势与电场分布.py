import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Parameters for the electric field
q1 = 10.0
a1 = 1.0
q2 = 10.0
a2 = 3.0
theta = np.pi / 3
eps = 1e-6

def U(x, y):
    return -q1 / np.sqrt((x + a1) ** 2 + y ** 2 + eps) \
           + q1 / np.sqrt((x - a1) ** 2 + y ** 2 + eps) \
           - q2 / np.sqrt((x + a2 * np.cos(theta)) ** 2 + (y + a2 * np.sin(theta)) ** 2 + eps) \
           + q2 / np.sqrt((x - a2 * np.cos(theta)) ** 2 + (y - a2 * np.sin(theta)) ** 2 + eps)

dx = 0.05
dy = 0.05
xaixs = np.arange(-5, 5, dx)
yaixs = np.arange(-5, 5, dy)

# 1. 创建网格：Meshgrid 会直接生成符合 Matplotlib 要求的 (Y, X) 维度矩阵
# 这样后续所有的计算都不再需要转置 (.T)
X, Y = np.meshgrid(xaixs, yaixs)

# 2. 向量化计算：直接传入矩阵，瞬间计算出整个平面的电势
apotential = U(X, Y)

# 3. 向量化计算电场分量：同样直接传入矩阵计算
Ex = -(U(X+dx, Y) - U(X-dx, Y)) / (2*dx)
Ey = -(U(X, Y+dy) - U(X, Y-dy)) / (2*dy)
afield = np.hypot(Ex, Ey)

# 限制极端值，避免奇点附近的数值主导整体颜色层次
vmax = np.percentile(np.abs(apotential), 98)
vmax = max(vmax, 1.0)
levels = np.linspace(-vmax, vmax, 41)

# --- 绘图部分 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(13, 6), constrained_layout=True)

# ========== 仅修改此部分：升级为精美等高线图，其余代码完全保留 ==========
# 电势等高线图
ax1 = fig.add_subplot(1, 2, 1)
extent = [-5.0, 5.0, -5.0, 5.0]

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cs_fill = ax1.contourf(X, Y, apotential, levels=levels, cmap='RdBu_r', norm=norm, alpha=0.95)
cs_main = ax1.contour(X, Y, apotential, levels=np.linspace(-vmax, vmax, 17), colors='k', linewidths=0.6, alpha=0.65)
ax1.clabel(cs_main, inline=True, fontsize=8, fmt='%.1f')
cb1 = fig.colorbar(cs_fill, ax=ax1, shrink=0.9, pad=0.02)
cb1.set_label('Electric Potential (a.u.)', fontsize=11)

charge_points = [
    (-a1, 0.0, '-q1'),
    (a1, 0.0, '+q1'),
    (-a2 * np.cos(theta), -a2 * np.sin(theta), '-q2'),
    (a2 * np.cos(theta), a2 * np.sin(theta), '+q2'),
]
for cx, cy, lab in charge_points:
    if lab.startswith('+'):
        ax1.scatter(cx, cy, s=60, c='#b2182b', edgecolors='white', linewidths=0.7, zorder=5)
    else:
        ax1.scatter(cx, cy, s=60, c='#2166ac', edgecolors='white', linewidths=0.7, zorder=5)
    ax1.text(cx + 0.08, cy + 0.08, lab, fontsize=9, weight='bold')
# ==============================================================================

# 电场图（完全保留原有代码）
ax2 = fig.add_subplot(1, 2, 2)
color = np.log10(afield + 1e-8)
stream = ax2.streamplot(
    X, Y, Ex, Ey,
    color=color,
    density=1.7,
    linewidth=1.0,
    arrowsize=1.1,
    cmap='magma'
)
cb2 = fig.colorbar(stream.lines, ax=ax2, shrink=0.9, pad=0.02)
cb2.set_label('log10|E| (a.u.)', fontsize=11)

# 坐标轴设置（完全保留原有代码）
ax1.set_xlabel('x', size=20)
ax1.set_ylabel('y', size=20)
ax2.set_xlabel('x', size=20)
ax2.set_ylabel('y', size=20)
ax1.set_xlim(-5.0, 5.0)
ax1.set_ylim(-5.0, 5.0)
ax2.set_xlim(-5.0, 5.0)
ax2.set_ylim(-5.0, 5.0)
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
ax1.set_title('Electric Potential Contours', size=16, pad=10)
ax2.set_title('Electric Field Streamlines', size=16, pad=10)
ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)

plt.show()