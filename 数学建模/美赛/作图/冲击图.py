import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
samples = np.array([1, 2, 3, 4, 5, 6])
# 每一层的高度数据 (4个类别，6个样本)
data = np.array([
    [0.19, 0.31, 0.25, 0.25, 0.16, 0.19],  # A
    [0.22, 0.35, 0.38, 0.28, 0.17, 0.22],  # B
    [0.23, 0.32, 0.33, 0.25, 0.25, 0.23],  # C
    [0.37, 0.37, 0.36, 0.32, 0.38, 0.37]  # D
])
labels = ['A', 'B', 'C', 'D']
# 马卡龙色系配色
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']

# 2. 计算堆叠的基准线 (Bottom)
cum_data = np.cumsum(data, axis=0)
# 在最下面补一行0，方便后续计算每一层的上下界
cum_data_with_zero = np.vstack([np.zeros(data.shape[1]), cum_data])

# 3. 设置参数
bar_width = 0.4  # 柱子的宽度
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 4. 开始绘图
for i in range(len(data)):
    # 每一层的颜色和上下界
    color = colors[i]
    y_lower = cum_data_with_zero[i]
    y_upper = cum_data_with_zero[i + 1]

    # --- 绘制离散的柱子 ---
    ax.bar(samples, data[i], bottom=y_lower, width=bar_width,
           color=color, edgecolor='black', linewidth=0.6, zorder=3)

    # --- 绘制柱子之间的填充过渡区域 ---
    # 我们需要在每个样本点之间绘制一个多边形填充
    for j in range(len(samples) - 1):
        x_left = samples[j] + bar_width / 2
        x_right = samples[j + 1] - bar_width / 2

        # 填充区域的四个顶点 (梯形)
        fill_x = [x_left, x_right, x_right, x_left]
        fill_y = [y_lower[j], y_lower[j + 1], y_upper[j + 1], y_upper[j]]

        # 使用 fill 绘制过渡区，alpha 设置稍透明增加质感
        ax.fill(fill_x, fill_y, color=color, alpha=0.7, edgecolor='none', zorder=2)

# 5. 图表美化
# 设置标题和标签
ax.set_title('Filled stacked bar chart', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Samples', fontsize=14)
ax.set_ylabel('RMSE (m)', fontsize=14)

# 设置坐标轴范围和刻度
ax.set_ylim(0, 1.5)
ax.set_xticks(samples)
ax.set_yticks(np.arange(0, 1.6, 0.3))

# 设置网格线（仅水平，置于底层）
ax.grid(axis='y', linestyle='-', linewidth=1.5, color='#e0e0e0', zorder=0)

# 隐藏上方和右侧的边框，模仿原图风格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 6. 自定义图例 (Legend)
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i])
                   for i in range(len(labels))]
ax.legend(handles=legend_elements[::-1], loc='upper right', frameon=True,
          edgecolor='black', fontsize=12, borderpad=0.5)

# 7. 调整布局并保存
plt.tight_layout()
# plt.savefig('filled_stacked_bar.pdf', bbox_inches='tight') # 建议存为PDF以保证论文清晰度
plt.show()