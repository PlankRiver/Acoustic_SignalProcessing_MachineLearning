import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（替换成你的真实数据）
# ----------------------
# X轴：月份
months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
n_months = len(months)

# 每个柱子的起始值和高度（对应图中的最低温和温差）
start_values = np.array([-12, -10, -5, -3, 0, 3, 9, 7, 4, -5, -7, -15])
heights = np.array([20, 15, 14, 21, 21, 25, 19, 18, 12, 15, 16, 23])

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(14, 10))

# ----------------------
# 3. 绘制悬浮柱状图
# ----------------------
bars = ax.bar(
    months, heights,
    bottom=start_values,
    color='black',
    edgecolor='black',
    alpha=0.9
)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加水平网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 设置坐标轴
ax.set_xlabel('Month', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylim(-16, 31)
ax.set_xticks(np.arange(n_months))
ax.set_xticklabels(months, fontsize=12, fontweight='bold')

# 添加标题
ax.set_title('Floating bar chart', fontsize=18, fontweight='bold', pad=20)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()