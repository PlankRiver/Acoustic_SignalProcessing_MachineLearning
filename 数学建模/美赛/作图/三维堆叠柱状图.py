import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# 1. 设置全局字体（学术论文风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备模拟数据 (10x10 的网格)
x_size, y_size = 10, 10
x = np.arange(1, x_size + 1)
y = np.arange(1, y_size + 1)
_x, _y = np.meshgrid(x, y)
x_flat, y_flat = _x.ravel(), _y.ravel()

# 设置 5 层采样数据 (Samp1-5) 的高度
# 我们生成一个中心对称分布的模拟数据
n_layers = 5
data = []
for i in range(n_layers):
    # 生成一些随机且具有分布感的数据
    z = (np.exp(-((x_flat-5)**2 + (y_flat-5)**2) / 20) * (10 - i*1.5)) + np.random.rand(100) * 2
    data.append(z)

# 3. 设置配色 (还原原图的蓝色渐变色系)
# 从深蓝到浅蓝
colors = ['#0000CD', '#1E90FF', '#4169E1', '#6495ED', '#87CEFA']
labels = [f'Samp{i+1}' for i in range(n_layers)]

# 4. 创建 3D 画布
fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 5. 核心绘图逻辑：堆叠绘制
dx = dy = 0.6  # 柱子的粗细
bottom = np.zeros_like(x_flat,dtype=float)  # 每一层堆叠的底部起始高度

for i in range(n_layers):
    # bar3d 参数: x, y, z(起点), dx, dy, dz(高度)
    ax.bar3d(x_flat - dx/2, y_flat - dy/2, bottom, dx, dy, data[i],
             color=colors[i], edgecolor='black', linewidth=0.3, shade=True)
    # 更新下一层的底部高度
    bottom += data[i]

# 6. 图表美化与细节还原
ax.set_title('Bar3Stack Plot', fontsize=16, fontweight='bold')
ax.set_xlabel('Variable1', fontsize=12)
ax.set_ylabel('Variable2', fontsize=12)
ax.set_zlabel('Variable3', fontsize=12)

# 设置坐标轴范围与刻度
ax.set_zlim(0, 60)
ax.set_xticks(x)
ax.set_yticks(y)

# 调整视角 (还原原图的俯视角度)
ax.view_init(elev=30, azim=45)

# 强制设置背景和网格为白色/灰色线（类似原图）
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')
ax.grid(True, linestyle='--', alpha=0.6)

# 7. 添加图例
legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=labels[i])
                   for i in range(n_layers)]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 0.8),
          frameon=True, edgecolor='black', fontsize=11)

# 8. 自动调整并显示
plt.tight_layout()
# plt.savefig('3d_stacked_bar.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()