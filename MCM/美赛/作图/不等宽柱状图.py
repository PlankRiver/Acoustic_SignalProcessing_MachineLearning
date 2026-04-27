import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局字体（美赛建议使用 Times New Roman 或类似学术字体）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据
labels = ['A', 'B', 'C', 'D', 'E']
heights = [46, 27, 53, 44, 55]  # 对应 RMSE (m) 的数值
widths = [19, 29, 18, 12, 12]   # 对应 Scale 的区间宽度
# 计算每根柱子的起始 x 坐标 (左边界)
# 第一个从0开始，后面的位置是前面宽度的累加
left_positions = [0] + list(np.cumsum(widths)[:-1])

# 3. 设置颜色（参考原图的马卡龙色系/Set3色板）
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

# 4. 绘图
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# 使用 bar 函数绘制，align='edge' 表示从指定左坐标向右画 width 宽度
bars = ax.bar(left_positions, heights, width=widths,
              color=colors, edgecolor='black', linewidth=1.2,
              align='edge', alpha=0.9)

# 5. 美化坐标轴
ax.set_xlabel('Scale', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE (m)', fontsize=12, fontweight='bold')
ax.set_title('Bar chart with unequal width', fontsize=14, fontweight='bold', pad=15)

# 设置坐标轴范围
ax.set_xlim(-2, 95)
ax.set_ylim(0, 60)

# 设置刻度
ax.set_xticks(np.arange(0, 100, 10))
ax.set_yticks(np.arange(0, 70, 10))

# 6. 添加网格线 (置于底层)
ax.grid(axis='both', linestyle='-', alpha=0.3, zorder=0)
ax.set_axisbelow(True) # 确保网格线在柱子下面

# 7. 添加图例
# 我们手动创建一个图例，匹配颜色和标签
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, edgecolor='black', fontsize=10)

# 8. 调整布局并显示/保存
plt.tight_layout()
# plt.savefig('my_chart.png', dpi=300, bbox_inches='tight') # 建议保存为高分辨率图片用于论文
plt.show()