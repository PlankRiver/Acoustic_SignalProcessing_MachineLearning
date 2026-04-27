import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局字体（美赛论文强制要求：英文使用 Times New Roman）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 准备数据 (根据原图近似取值)
x = [20, 40, 60, 80, 100, 120]
y_k10 = [0.98, 0.96, 0.88, 0.81, 0.73, 0.69]
y_k20 = [0.98, 0.92, 0.80, 0.70, 0.65, 0.61]
y_k30 = [0.98, 0.87, 0.75, 0.63, 0.52, 0.43]

# 3. 创建画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# 4. 绘图 (还原原图配色与标记)
# markerfacecolor='none' 实现空心标记
ax.plot(x, y_k10, label='k=10', color='red', linewidth=2,
        marker='o', markersize=10, markerfacecolor='none', markeredgewidth=2)

ax.plot(x, y_k20, label='k=20', color='blue', linewidth=2,
        marker='*', markersize=12)

ax.plot(x, y_k30, label='k=30', color='lime', linewidth=2,
        marker='^', markersize=10, markerfacecolor='none', markeredgewidth=2)

# 5. 美化细节
# 设置坐标轴标签 (美赛请务必使用英文：Count / Percentage)
ax.set_xlabel('Count (个数)', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (百分比)', fontsize=14, fontweight='bold')

# 设置轴范围和刻度
ax.set_xlim(20, 125)
ax.set_ylim(0, 1.05)
ax.set_xticks(np.arange(20, 121, 20))
ax.set_yticks(np.arange(0, 1.1, 0.2))

# 开启网格线 (灰色，较淡)
ax.grid(True, linestyle='-', alpha=0.3)

# 6. 图例设置 (放在右上角，设置边框)
ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)

# 7. 设置坐标轴线粗细
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 8. 调整布局并保存
plt.tight_layout()
# plt.savefig('line_chart.pdf', bbox_inches='tight') # 建议保存为PDF格式
plt.show()