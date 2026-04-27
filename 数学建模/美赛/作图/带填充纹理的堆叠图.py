import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (13个样本, 4个类别)
samples = np.arange(1, 14)
# 模拟数据：A, B, C, D 四组的高度
data_A = np.array([0.19, 0.31, 0.25, 0.25, 0.16, 0.19, 0.31, 0.25, 0.16, 0.31, 0.25, 0.25, 0.16])
data_B = np.array([0.22, 0.35, 0.38, 0.28, 0.17, 0.22, 0.35, 0.28, 0.17, 0.35, 0.38, 0.28, 0.17])
data_C = np.array([0.23, 0.32, 0.33, 0.25, 0.25, 0.23, 0.32, 0.25, 0.25, 0.32, 0.33, 0.25, 0.25])
data_D = np.array([0.37, 0.37, 0.36, 0.32, 0.38, 0.37, 0.37, 0.32, 0.38, 0.37, 0.36, 0.32, 0.38])

labels = ['A', 'B', 'C', 'D']
# 对应颜色 (黄色, 浅绿, 紫色, 嫩绿)
colors = ['#f9e05e', '#c4e8c1', '#af77b5', '#a3d14f']
# 对应纹理 (关键点)
# 'xx': 小网格, '//': 右斜线, '--': 水平线, '\\\\': 左斜线
hatches = ['xx', '//', '--', '\\\\']

# 2. 计算堆叠位置
bottom_B = data_A
bottom_C = data_A + data_B
bottom_D = data_A + data_B + data_C

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 依次绘制每一层，通过 hatch 参数添加纹理
# edgecolor 必须设为黑色或深色，纹理才清晰
ax.bar(samples, data_A, label='A', color=colors[0], hatch=hatches[0], edgecolor='black', linewidth=0.8)
ax.bar(samples, data_B, bottom=bottom_B, label='B', color=colors[1], hatch=hatches[1], edgecolor='black', linewidth=0.8)
ax.bar(samples, data_C, bottom=bottom_C, label='C', color=colors[2], hatch=hatches[2], edgecolor='black', linewidth=0.8)
ax.bar(samples, data_D, bottom=bottom_D, label='D', color=colors[3], hatch=hatches[3], edgecolor='black', linewidth=0.8)

# 4. 图表美化
ax.set_title('Texture filled stacked bar chart', fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel('Samples', fontsize=14)
ax.set_ylabel('RMSE (m)', fontsize=14)

# 坐标轴刻度设置
ax.set_ylim(0, 1.5)
ax.set_xticks(samples)
ax.set_yticks(np.arange(0, 1.6, 0.3))

# 网格线：仅水平，淡灰色，置于底层
ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', zorder=0)
ax.set_axisbelow(True)

# 隐藏上方和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 5. 图例设置 (包含纹理展示)
ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)

# 6. 调整布局并保存
plt.tight_layout()
# plt.savefig('texture_stacked_bar.png', dpi=300)
plt.show()