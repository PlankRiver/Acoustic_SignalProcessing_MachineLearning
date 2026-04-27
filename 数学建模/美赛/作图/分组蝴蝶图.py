import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
samples = ['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5', 'Sample6', 'Sample7', 'Sample8']
# 为了让绘图从上往下排列，我们需要反转列表
samples = samples[::-1]

# 左侧数据 (Feature 1 & 2)
f1 = [230, 250, 260, 360, 400, 430, 490, 580]
f2 = [70, 80, 110, 130, 220, 260, 290, 300]

# 右侧数据 (Feature 3 & 4)
f3 = [330, 460, 520, 550, 600, 640, 710, 740]
f4 = [130, 200, 230, 240, 390, 400, 420, 460]

# 2. 设置参数
y = np.arange(len(samples))  # 产生纵坐标位置
height = 0.35  # 单个条形的高度

# 配色方案 (马卡龙色系)
colors = ['#80b1d3', '#fb8072', '#fdb462', '#b3de69'] # 蓝、红、橙、绿

# 3. 创建画布，设置左右两个子图，共享y轴
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 8), dpi=150, sharey=True)

# --- 绘制左侧图表 (向左延伸) ---
# Feature 1 放在上面，Feature 2 放在下面
ax_l.barh(y + height/2, f1, height, color=colors[0], edgecolor='black', linewidth=0.7, label='Feature1')
ax_l.barh(y - height/2, f2, height, color=colors[1], edgecolor='black', linewidth=0.7, label='Feature2')

# --- 绘制右侧图表 (向右延伸) ---
ax_r.barh(y + height/2, f3, height, color=colors[2], edgecolor='black', linewidth=0.7, label='Feature3')
ax_r.barh(y - height/2, f4, height, color=colors[3], edgecolor='black', linewidth=0.7, label='Feature4')

# 4. 美化坐标轴
# 左侧轴设置
ax_l.invert_xaxis()  # 翻转X轴方向
ax_l.set_xlim(600, 0)
ax_l.set_xticks(np.arange(0, 601, 100))

# 右侧轴设置
ax_r.set_xlim(0, 800)
ax_r.set_xticks(np.arange(0, 801, 200))

# 隐藏不需要的边框，模仿原图简洁风格
for ax in [ax_l, ax_r]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 添加垂直背景网格
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

# 5. 设置中间的文字标签
# 调整两个子图的间距
plt.subplots_adjust(wspace=0.22)
for i, sample in enumerate(samples):
    # 将样本名称放在两个子图中间的空白处
    # 通过 ax_r 的 text 功能，设置横坐标为负值来实现居中
    ax_r.text(-90, i, sample, ha='center', va='center', fontsize=12, fontfamily='serif')

# 6. 设置图例 (放在顶部)
ax_l.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=11)
ax_r.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=11)

# 7. 调整布局并显示
plt.tight_layout()
# plt.savefig('grouped_butterfly_chart.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()