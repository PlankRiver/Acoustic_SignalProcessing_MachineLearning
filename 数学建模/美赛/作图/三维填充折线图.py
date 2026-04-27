import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


# 1. 模拟数据生成
def generate_data():
    x = np.linspace(0, 100, 200)  # Mass/Charge (M/Z)
    samples = ['Samp A', 'Samp B', 'Samp C', 'Samp D', 'Samp E']
    zs = np.arange(len(samples))  # 样本对应的 Y 轴位置

    verts = []
    for i in range(len(samples)):
        # 生成带噪声的波峰数据
        y = np.exp(-((x - (20 + i * 15)) ** 2) / 100) + np.random.normal(0, 0.02, len(x))
        y[y < 0] = 0
        # 为了填充，需要在曲线两端添加起点和终点到 y=0 的路径
        y[0], y[-1] = 0, 0
        verts.append(list(zip(x, y)))

    return x, zs, verts, samples


x, zs, verts, sample_labels = generate_data()

# 2. 绘图设置
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 设置颜色（与原图接近的明亮色系）
colors = ['#7fcdbb', '#ffff99', '#bebada', '#fb8072', '#80b1d3']

# 3. 核心：创建填充集合
poly = PolyCollection(verts, facecolors=colors, alpha=0.7, edgecolors='gray', linewidths=0.5)
ax.add_collection3d(poly, zs=zs, zdir='y')

# 4. 细节微调
ax.set_xlabel('Mass/Charge (M/Z)', labelpad=10)
ax.set_ylabel('Samples', labelpad=10)
ax.set_zlabel('Ion Spectra', labelpad=10)

# 设置坐标轴范围
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, len(sample_labels) - 0.5)
ax.set_zlim(0, 1.5)

# 设置 Y 轴刻度标签为样本名
ax.set_yticks(zs)
ax.set_yticklabels(sample_labels)

# 调整视角以匹配原图
ax.view_init(elev=30, azim=-50)

# 去掉背景网格的灰色填充（可选，使图片更清爽）
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.title('Extracted Spectra Subset', fontweight='bold')
plt.tight_layout()
plt.show()