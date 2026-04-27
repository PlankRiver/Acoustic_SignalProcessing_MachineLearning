import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# ----------------------
# 1. 生成模拟数据（替换成你的真实数据）
# ----------------------
np.random.seed(42)
n_points = 5000

# X, Y, Z 坐标数据
x = np.random.uniform(85, 90, n_points)
y = np.random.uniform(45, 50, n_points)
z = 78 + 14 * ( (x-85)/5 )**2 + ( (y-45)/5 )**2 + np.random.normal(0, 0.5, n_points)

# 两个特征值（用于颜色渲染）
feature1 = z + np.random.normal(0, 1, n_points)  # 右上角颜色条
feature2 = 77 + 3 * ( (x-85)/5 ) + 3 * ( (y-45)/5 ) + np.random.normal(0, 0.5, n_points)  # 右下角颜色条

# ----------------------
# 2. 绘图设置
# ----------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# ----------------------
# 3. 绘制双特征散点图
# ----------------------
# 第一组散点（对应右上角颜色条：红-橙-白-蓝）
scatter1 = ax.scatter(x, y, z, c=feature1, cmap=cm.coolwarm, s=15, alpha=0.8)
# 第二组散点（对应右下角颜色条：白-绿-蓝-黑）
scatter2 = ax.scatter(x, y, 78 + (z-78)*0.2, c=feature2, cmap=cm.viridis_r, s=15, alpha=0.6)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
ax.set_title('DoubleFeatureScatter Plot', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_zlabel('Z', fontsize=12, fontweight='bold')

# 添加两个颜色条
cbar1 = fig.colorbar(scatter1, ax=ax, shrink=0.8, aspect=15, pad=0.05)
cbar1.set_label('Feature 1 Value', fontsize=10, fontweight='bold')
cbar2 = fig.colorbar(scatter2, ax=ax, shrink=0.8, aspect=15, pad=0.15)
cbar2.set_label('Feature 2 Value', fontsize=10, fontweight='bold')

# 设置视角
ax.view_init(elev=30, azim=-60)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 显示图表（不保存PDF）
# ----------------------
plt.show()