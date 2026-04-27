import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（替换成你的真实数据）
# ----------------------
# Y轴类别
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
n_categories = len(categories)

# 左侧（负数）堆叠数据
left_feature1 = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.4])
left_feature2 = np.array([-0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.4])
left_feature3 = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3])
left_feature4 = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3])
left_feature5 = np.array([-0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3])
left_feature6 = np.array([-0.1, -0.1, -0.2, -0.2, -0.3, -0.3, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4])

# 右侧（正数）堆叠数据
right_feature1 = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4])
right_feature2 = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4])
right_feature3 = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
right_feature4 = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
right_feature5 = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
right_feature6 = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(14, 10))

# 配色（和原图一致）
colors = ['#7BC950', '#FFF275', '#B19CD9', '#FF6961', '#6A93D4', '#FFA62B']
labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6']

# ----------------------
# 3. 绘制水平双向堆叠柱状图
# ----------------------
# 绘制左侧（负数）堆叠
ax.barh(categories, left_feature1, color=colors[0], label=labels[0])
ax.barh(categories, left_feature2, left=left_feature1, color=colors[1], label=labels[1])
ax.barh(categories, left_feature3, left=left_feature1+left_feature2, color=colors[2], label=labels[2])
ax.barh(categories, left_feature4, left=left_feature1+left_feature2+left_feature3, color=colors[3], label=labels[3])
ax.barh(categories, left_feature5, left=left_feature1+left_feature2+left_feature3+left_feature4, color=colors[4], label=labels[4])
ax.barh(categories, left_feature6, left=left_feature1+left_feature2+left_feature3+left_feature4+left_feature5, color=colors[5], label=labels[5])

# 绘制右侧（正数）堆叠
ax.barh(categories, right_feature1, color=colors[0])
ax.barh(categories, right_feature2, left=right_feature1, color=colors[1])
ax.barh(categories, right_feature3, left=right_feature1+right_feature2, color=colors[2])
ax.barh(categories, right_feature4, left=right_feature1+right_feature2+right_feature3, color=colors[3])
ax.barh(categories, right_feature5, left=right_feature1+right_feature2+right_feature3+right_feature4, color=colors[4])
ax.barh(categories, right_feature6, left=right_feature1+right_feature2+right_feature3+right_feature4+right_feature5, color=colors[5])

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加中心基准线
ax.axvline(x=0, color='black', linewidth=1.2, linestyle='--')

# 设置坐标轴
ax.set_xlabel('X-axis', fontsize=12, fontweight='bold')
ax.set_ylabel('Y-axis', fontsize=12, fontweight='bold')
ax.set_xlim(-4, 5)
ax.set_yticks(np.arange(n_categories))
ax.set_yticklabels(categories, fontsize=11)

# 添加标题和图例
ax.set_title('Stacked Bidirectional bar chart', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 显示图表
# ----------------------
plt.show()