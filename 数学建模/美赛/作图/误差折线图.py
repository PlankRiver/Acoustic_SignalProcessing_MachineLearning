import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备（替换成你的真实数据）
# ----------------------
# 数值横坐标版本（对应第一张图）
x_numeric = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
y_values = np.array([20, 30, 45, 40, 58, 64, 80, 74, 95, 90])
y_errors = np.array([8, 7, 8, 9, 7, 8, 8, 9, 10, 9])

# 分类横坐标版本（对应后面几张图）
x_categories = ['Tink', 'Malx', 'KNC', 'BXY', 'CNN', 'ALL', 'TXTP', 'WNET', 'SUY', 'MAG', 'CTI']
y_categorical = np.array([0.6, 0.8, 0.4, 0.5, 0.55, 0.6, 0.8, 0.9, 0.93, 0.82, 0.75])
y_cat_errors = np.array([0.08, 0.09, 0.07, 0.06, 0.07, 0.08, 0.09, 0.08, 0.1, 0.08, 0.09])

# ----------------------
# 2. 绘图设置
# ----------------------
# 可以切换两种横坐标模式
use_categorical_x = False  # True: 分类横坐标 / False: 数值横坐标

if use_categorical_x:
    x = x_categories
    y = y_categorical
    y_err = y_cat_errors
else:
    x = x_numeric
    y = y_values
    y_err = y_errors

# 专业配色（美赛常用）
color = '#E63946'  # 珊瑚红
marker_style = 'o'  # 可选 'o', '^', '*', 's' 等
line_style = '-'

# ----------------------
# 3. 绘制误差折线图
# ----------------------
fig, ax = plt.subplots(figsize=(12, 9))

# 绘制带误差棒的折线
ax.errorbar(
    x, y, yerr=y_err, fmt=marker_style, color=color, ecolor=color,
    elinewidth=2, capsize=5, markersize=8, linestyle=line_style,
    linewidth=2, alpha=0.8
)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 添加网格线（可选）
ax.grid(True, linestyle='--', alpha=0.3)

# 设置坐标轴
ax.set_xlabel('横坐标', fontsize=12, fontweight='bold')
ax.set_ylabel('实验结果', fontsize=12, fontweight='bold')
if use_categorical_x:
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(0.3, 1.1)
else:
    ax.set_ylim(0, 110)

# 添加标题
ax.set_title('误差折线图', fontsize=16, fontweight='bold', pad=20)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 显示图表
# ----------------------
plt.show()