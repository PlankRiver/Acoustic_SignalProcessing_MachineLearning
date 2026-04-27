import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. 准备模拟数据
# 16个方位
directions = np.radians(np.linspace(0, 360, 16, endpoint=False))
n_directions = len(directions)
n_categories = 7

# 模拟数据 (行:方位, 列:风速等级)
np.random.seed(42) # 固定随机种子方便复现
data = np.random.rand(n_directions, n_categories) * 0.8
data = data / data.sum(axis=1)[:, None] * np.random.uniform(2, 5, n_directions)[:, None]

# 2. 设置颜色方案 (深度还原原图的 绿-白-棕 渐变)
colors = ['#4d2906', '#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571', '#003c30']
# 颜色反转：让深绿色在最外圈
colors = colors[::-1]

# 3. 创建极坐标画布
fig = plt.figure(figsize=(12, 10), dpi=150)
ax = fig.add_subplot(111, projection='polar')

# --- 修正后的位置：这里删掉了多余的 theta ---
ax.set_theta_zero_location('N') # 0度(北)在上方
ax.set_theta_direction(-1)      # 顺时针排列

# 4. 核心绘图逻辑：循环堆叠绘制
width = 2 * np.pi / n_directions * 0.8 # 柱子宽度
offset = 0.5                           # 中心圆孔大小
bottom = np.zeros(n_directions) + offset

for i in range(n_categories):
    ax.bar(directions, data[:, i], width=width, bottom=bottom,
           color=colors[i], edgecolor='black', linewidth=0.5)
    bottom += data[:, i]

# 5. 图表美化与细节还原
ax.set_title('Wind Rose Plot', fontsize=18, fontweight='bold', pad=30)

# 设置方位标签 (N, NE, E ...)
ax.set_thetagrids(np.linspace(0, 360, 8, endpoint=False),
                  labels=['N (0°)', 'NE (45°)', 'E (90°)', 'SE (135°)',
                          'S (180°)', 'SW (225°)', 'W (270°)', 'NW (315°)'],
                  fontsize=13)

# 设置径向刻度 (百分比)
ax.set_rlabel_position(155)
ax.set_rticks([1.5, 2.5, 3.5, 4.5, 5.5])
ax.set_yticklabels(['0.9%', '1.8%', '2.7%', '3.6%', '4.5%'], fontsize=12)

# 在中心圆圈写上 0%
ax.text(0, 0, "0%", ha='center', va='center', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='-', alpha=0.5)

# 6. 自定义图例
legend_labels = [
    '$W_s \geq 18$',
    '$15 \leq W_s < 18$',
    '$12 \leq W_s < 15$',
    '$9 \leq W_s < 12$',
    '$6 \leq W_s < 9$',
    '$3 \leq W_s < 6$',
    '$0 \leq W_s < 3$'
]

# 注意顺序：从外往内
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=legend_labels[i])
                   for i in range(n_categories)]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.25, 1.05),
          title='Wind speeds in m/s', title_fontsize=14, frameon=False, fontsize=12)

# 7. 调整布局并显示
plt.tight_layout()
plt.show()