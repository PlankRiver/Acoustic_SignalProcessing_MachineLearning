import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（替换成你的真实数据）
# ----------------------
# X轴类别
categories = ['Masculine', 'Agency', 'Feminine', 'Communion']
# 两组数据（Odd Numbers / Even Numbers）
odd_data = [0.24, 0.22, 0.24, 0.19]
even_data = [0.33, 0.25, 0.26, 0.33]
# 误差棒数据（标准误或标准差）
odd_error = [0.03, 0.02, 0.03, 0.02]
even_error = [0.04, 0.03, 0.03, 0.04]

# ----------------------
# 2. 绘图设置
# ----------------------
# 专业配色（美赛常用）
colors = ['#FFF1D0', '#1D3557']  # 浅米黄 + 深蓝色
# 柱子宽度
bar_width = 0.35
# X轴位置
x = np.arange(len(categories))

# ----------------------
# 3. 绘制图表
# ----------------------
fig, ax = plt.subplots(figsize=(10, 7))

# 绘制两组柱状图
bars1 = ax.bar(x - bar_width/2, odd_data, bar_width, label='Odd Numbers', color=colors[0], edgecolor='black', yerr=odd_error, capsize=5)
bars2 = ax.bar(x + bar_width/2, even_data, bar_width, label='Even Numbers', color=colors[1], edgecolor='black', yerr=even_error, capsize=5)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加基准虚线
ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1.5, label='Chance Level')

# 设置坐标轴
ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
ax.set_xlabel('Category', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 0.5)

# 添加标题和图例
ax.set_title('Accuracy by Category and Number Type', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 保存或显示
# ----------------------
# 保存为高清图片（美赛建议用PDF或300dpi的PNG）
# plt.savefig('grouped_bar_chart.pdf', dpi=300, bbox_inches='tight')
plt.show()