import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
samples = ['samp1', 'samp2', 'samp3', 'samp4']
# 每一类在四个样本下的数值 (近似原图数据)
data_A = [0.44, 0.32, 0.44, 0.20]
data_B = [0.49, 0.35, 0.49, 0.33]
data_C = [0.49, 0.34, 0.47, 0.29]
data_D = [0.47, 0.34, 0.45, 0.24]

labels = ['A', 'B', 'C', 'D']
# 配色方案 (薄荷绿, 浅黄, 浅紫, 橙色)
colors = ['#7bcfa6', '#fdfd96', '#c7c7e2', '#ffb347']
# 填充纹理 (xx: 交叉斜线, //: 右斜线, --: 水平线, \\: 左斜线)
hatches = ['xx', '//', '--', '\\\\']

# 2. 计算柱子位置 (分组柱状图的关键)
x = np.arange(len(samples))  # 样本组的中心位置
width = 0.18  # 单个柱子的宽度

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 依次绘制每一类柱子，通过偏移量 width 分开
# zorder=3 确保柱子在网格线上方
rects1 = ax.bar(x - 1.5*width, data_A, width, label='A', color=colors[0],
                hatch=hatches[0], edgecolor='black', linewidth=1.2, zorder=3)
rects2 = ax.bar(x - 0.5*width, data_B, width, label='B', color=colors[1],
                hatch=hatches[1], edgecolor='black', linewidth=1.2, zorder=3)
rects3 = ax.bar(x + 0.5*width, data_C, width, label='C', color=colors[2],
                hatch=hatches[2], edgecolor='black', linewidth=1.2, zorder=3)
rects4 = ax.bar(x + 1.5*width, data_D, width, label='D', color=colors[3],
                hatch=hatches[3], edgecolor='black', linewidth=1.2, zorder=3)

# 4. 图表美化
ax.set_title('Texture filled bar chart', fontsize=20, fontweight='bold', pad=15)
ax.set_ylabel('RMSE (m)', fontsize=16)
ax.set_xlabel('Samples', fontsize=16)

# 设置坐标轴范围和刻度
ax.set_ylim(0, 0.6)
ax.set_xticks(x)
ax.set_xticklabels(samples, fontsize=14)
ax.set_yticks(np.arange(0, 0.7, 0.1))

# 添加水平网格线 (置于底层)
ax.yaxis.grid(True, linestyle='-', color='#e0e0e0', linewidth=1.5, zorder=0)

# 移除上方和右侧边框 (美赛常用简洁风格)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 5. 图例设置
# 放在右上角，设置边框
ax.legend(loc='upper right', frameon=True, edgecolor='black',
          fontsize=14, facecolor='white', framealpha=1)

# 6. 调整布局并保存
plt.tight_layout()
# plt.savefig('texture_bar_comparison.png', dpi=300)
plt.show()