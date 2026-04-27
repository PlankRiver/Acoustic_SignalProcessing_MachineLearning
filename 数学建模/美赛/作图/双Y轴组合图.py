import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（示例数据，你可以替换成自己的）
# ----------------------
# X轴的类别（比如你的样本名称）
categories = [
    'sadhiusaduias', 'sadsafasag', 'egregreg', 'fgdgfgg3', 'wrttertrete',
    'werewrtg', 'dfdsfdsvfdg', 'dr34r43rf', 'werewtret', '454fgfgfg',
    'dsd3rff', 'rt56yhrgtfb', 'fret54rgtvfd', 'fwr33defdsf'
]
# 左侧Y轴：柱状图数据（Number）
bar_data = [0, 1000, 200, 150, 100, 4500, 1800, 6200, 10000, 0, 4000, 0, 300, 3000]
# 右侧Y轴：散点图数据（RC值）
scatter_data = [0.3, 0.28, 0.32, 0.3, 0.7, 0.68, 0.32, 0.42, 0.62, 0.46, 0.52, 0.3, 0.3, 0.52]
# 散点图的误差棒数据
error_data = [0.1, 0.12, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.18, 0.12, 0.15, 0.1, 0.1, 0.2]

# ----------------------
# 2. 绘图设置（美赛风格）
# ----------------------
# 创建画布和主坐标轴
fig, ax1 = plt.subplots(figsize=(16, 9))  # 宽屏比例，适合美赛报告
ax2 = ax1.twinx()  # 创建双Y轴

# 绘制柱状图（左侧Y轴）
bars = ax1.bar(categories, bar_data, color='#FFF1D0', edgecolor='black', alpha=0.8, label='Number')
# 绘制散点图+误差棒（右侧Y轴）
scatter = ax2.errorbar(
    categories, scatter_data, yerr=error_data, fmt='o', color='#E63946',
    capsize=5, elinewidth=2, markersize=8, label='RC'
)

# ----------------------
# 3. 美化图表（美赛必备）
# ----------------------
# 设置坐标轴标签
ax1.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of need', fontsize=12, fontweight='bold', color='#1D3557')
ax2.set_ylabel('$R_{TC}$', fontsize=12, fontweight='bold', color='#E63946')

# 设置坐标轴刻度颜色
ax1.tick_params(axis='y', labelcolor='#1D3557')
ax2.tick_params(axis='y', labelcolor='#E63946')

# 旋转X轴标签，防止重叠
plt.xticks(rotation=45, ha='right', fontsize=10)

# 设置标题
ax1.set_title('Dual Y-axis Combined Chart', fontsize=14, fontweight='bold', pad=20)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# 优化布局
plt.tight_layout()

# ----------------------
# 4. 保存或显示图表
# ----------------------
# 保存为高清图片（美赛建议用PDF或300dpi的PNG）
# plt.savefig('dual_axis_chart.pdf', dpi=300, bbox_inches='tight')
# 显示图表
plt.show()