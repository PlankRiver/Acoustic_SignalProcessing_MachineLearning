import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局样式（建议美赛使用 serif 字体，更具学术感）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据
months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.',
          'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']

# 这里的 lows 是柱子的底部，highs 是柱子的顶部
lows =  [-12.0, -10.1, -5.2, -2.1,  0.0,  2.6,  8.6,  7.4,  3.4, -4.1, -7.3, -14.5]
highs = [ 8.0,   5.5,   8.7,  18.5, 21.3, 28.6, 29.2, 25.7, 16.0, 10.2,  9.3,   8.5]

# 计算高度 (height = high - low)
heights = [h - l for h, l in zip(highs, lows)]

# 3. 设置配色（还原原图的多色马卡龙色系）
colors = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
    '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'
]

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 核心逻辑：使用 bar 函数，通过 bottom 参数指定柱子起始位置
bars = ax.bar(months, heights, bottom=lows, color=colors,
              width=0.7, zorder=3) # zorder确保柱子在网格线上方

# 5. 美化细节
ax.set_title('Floating bar chart', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Temperature($^\circ$C)', fontsize=13)
ax.set_xlabel('Month', fontsize=13)

# 设置坐标轴范围和刻度
ax.set_ylim(-15, 30)
ax.set_yticks(np.arange(-15, 31, 5))

# 添加水平网格线（淡灰色，置于底层）
ax.grid(axis='y', linestyle='-', linewidth=1, color='#e0e0e0', zorder=0)

# 调整边框厚度
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 设置刻度线向内还是向外
ax.tick_params(direction='out', length=5, width=1.2)

# 6. 自动调整布局并显示
plt.tight_layout()
# plt.savefig('floating_bar_chart.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()