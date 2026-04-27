import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据 (8个样本，每个样本左右各4个维度)
samples = ['Sample' + str(i) for i in range(1, 9)]
# 左侧数据 (F1-F4)
f1 = np.array([750, 600, 550, 480, 420, 350, 300, 220])
f2 = np.array([650, 550, 450, 380, 330, 280, 250, 150])
f3 = np.array([500, 450, 400, 300, 250, 200, 150, 100])
f4 = np.array([450, 350, 250, 200, 180, 150, 80,  60])

# 右侧数据 (F5-F8)
f5 = np.array([780, 590, 440, 410, 370, 350, 250, 230])
f6 = np.array([580, 450, 330, 280, 350, 250, 230, 120])
f7 = np.array([520, 480, 470, 310, 300, 120, 110, 100])
f8 = np.array([480, 450, 390, 400, 220, 200, 180, 90])

# 配色方案 (根据原图马卡龙色系)
colors_left = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']  # F1-F4
colors_right = ['#bc80bd', '#d9d9d9', '#fccde5', '#b3de69'] # F5-F8

# 2. 创建画布，设置两个并排的子图
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 8), dpi=150, sharey=True)

y_pos = np.arange(len(samples))
height = 0.7  # 条形高度

# 3. 绘制左侧堆叠图 (向左延伸)
# 原理：利用 barh 的 left 参数，并手动计算偏移
ax_l.barh(y_pos, f1, height, color=colors_left[0], edgecolor='black', linewidth=0.5, label='F1')
ax_l.barh(y_pos, f2, height, left=f1, color=colors_left[1], edgecolor='black', linewidth=0.5, label='F2')
ax_l.barh(y_pos, f3, height, left=f1+f2, color=colors_left[2], edgecolor='black', linewidth=0.5, label='F3')
ax_l.barh(y_pos, f4, height, left=f1+f2+f3, color=colors_left[3], edgecolor='black', linewidth=0.5, label='F4')

# 4. 绘制右侧堆叠图 (向右延伸)
ax_r.barh(y_pos, f5, height, color=colors_right[0], edgecolor='black', linewidth=0.5, label='F5')
ax_r.barh(y_pos, f6, height, left=f5, color=colors_right[1], edgecolor='black', linewidth=0.5, label='F6')
ax_r.barh(y_pos, f7, height, left=f5+f6, color=colors_right[2], edgecolor='black', linewidth=0.5, label='F7')
ax_r.barh(y_pos, f8, height, left=f5+f6+f7, color=colors_right[3], edgecolor='black', linewidth=0.5, label='F8')

# 5. 美化坐标轴
# 左侧轴处理：反转X轴，隐藏Y轴标签（移到中间）
ax_l.invert_xaxis()
ax_l.set_xlim(2500, 0) # 设置统一刻度
ax_l.set_xticks(np.arange(0, 2501, 500))

# 右侧轴处理
ax_r.set_xlim(0, 2500)
ax_r.set_xticks(np.arange(0, 2501, 500))

# 隐藏多余边框
for ax in [ax_l, ax_r]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3) # 垂直网格线
    ax.set_axisbelow(True)

# 6. 设置中间的 Sample 标签
# 技巧：通过调整两个子图的间距，并将标签放在 ax_r 的左侧
plt.subplots_adjust(wspace=0.25) # 留出中间位置
for i, sample in enumerate(samples):
    # 将文本放在两个轴的中央
    ax_r.text(-125, i, sample, ha='center', va='center', fontsize=12)

# 设置样本1在最上方
ax_l.invert_yaxis()

# 7. 添加图例 (放在顶部)
ax_l.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=11)
ax_r.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=11)

# 8. 保存与显示
plt.tight_layout()
# plt.savefig('butterfly_stacked.pdf', bbox_inches='tight')
plt.show()