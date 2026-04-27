import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（替换成你的真实数据）
# ----------------------
# X轴类别
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
n_categories = len(categories)

# 特征数据（正数部分）
feature1_pos = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
feature2_pos = np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.1])
feature3_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1])
feature4_pos = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1])
feature5_pos = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1])
feature6_pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1])

# 特征数据（负数部分）
feature1_neg = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
feature2_neg = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1, -0.1, -0.1])
feature3_neg = np.array([-0.1, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2, -0.1])
feature4_neg = np.array([-0.1, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2, -0.1])
feature5_neg = np.array([-0.1, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2, -0.1])
feature6_neg = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.7, -0.6, -0.5, -0.4, -0.3, -0.1])

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(14, 10))

# 配色（和原图一致）
colors = ['#7BC950', '#FFF275', '#B19CD9', '#FF6961', '#6A93D4', '#FFA62B']
labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6']

# ----------------------
# 3. 绘制双向堆叠柱状图
# ----------------------
# 绘制正数堆叠
ax.bar(categories, feature1_pos, color=colors[0], label=labels[0])
ax.bar(categories, feature2_pos, bottom=feature1_pos, color=colors[1], label=labels[1])
ax.bar(categories, feature3_pos, bottom=feature1_pos+feature2_pos, color=colors[2], label=labels[2])
ax.bar(categories, feature4_pos, bottom=feature1_pos+feature2_pos+feature3_pos, color=colors[3], label=labels[3])
ax.bar(categories, feature5_pos, bottom=feature1_pos+feature2_pos+feature3_pos+feature4_pos, color=colors[4], label=labels[4])
ax.bar(categories, feature6_pos, bottom=feature1_pos+feature2_pos+feature3_pos+feature4_pos+feature5_pos, color=colors[5], label=labels[5])

# 绘制负数堆叠
ax.bar(categories, feature1_neg, color=colors[0])
ax.bar(categories, feature2_neg, bottom=feature1_neg, color=colors[1])
ax.bar(categories, feature3_neg, bottom=feature1_neg+feature2_neg, color=colors[2])
ax.bar(categories, feature4_neg, bottom=feature1_neg+feature2_neg+feature3_neg, color=colors[3])
ax.bar(categories, feature5_neg, bottom=feature1_neg+feature2_neg+feature3_neg+feature4_neg, color=colors[4])
ax.bar(categories, feature6_neg, bottom=feature1_neg+feature2_neg+feature3_neg+feature4_neg+feature5_neg, color=colors[5])

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加基准线
ax.axhline(y=0, color='black', linewidth=1.2)

# 设置坐标轴
ax.set_ylabel('Y-axis', fontsize=12, fontweight='bold')
ax.set_xlabel('X-axis', fontsize=12, fontweight='bold')
ax.set_ylim(-3, 5)
ax.set_xticks(np.arange(n_categories))
ax.set_xticklabels(categories, fontsize=11)

# 添加标题和图例
ax.set_title('Stacked bidirectional bar chart', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 显示图表
# ----------------------
plt.show()