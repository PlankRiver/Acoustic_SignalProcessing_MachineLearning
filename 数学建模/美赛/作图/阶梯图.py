import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局学术字体 (美赛建议使用 Times New Roman 风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 准备数据
# x 为时间轴，y 为幅值
x = np.arange(21)
# 模拟一组随机的离散状态数据
np.random.seed(42)
y1 = np.random.randint(1, 8, size=21)
y2 = np.sin(x/5) * 4 + 4  # 模拟连续信号的阶梯化

# 3. 创建画布
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 4. 绘制阶梯图
# where='post' 表示在 x 处发生变化（最常用，对应原图效果）
# where='pre' 表示在 x 之前就发生变化
ax.step(x, y1, where='post', color='#2E3192', linewidth=2, label='System State A')

# 如果需要绘制多条线（如原图 2, 3, 6）
# 可以通过 linestyle='--' 设置虚线，设置不同颜色
# ax.step(x, y2, where='post', color='red', linestyle='--', linewidth=2, label='System State B')

# 5. 美化细节 (高度还原原图样式)
ax.set_title('Step Plot Template for MCM/ICM', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('time', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')

# 设置刻度范围和步长
ax.set_xlim(0, 20)
ax.set_ylim(1, 8)
ax.set_xticks(np.arange(0, 21, 5))
ax.set_yticks(np.arange(1, 9, 1))

# 设置网格 (原图特征：淡灰色网格)
ax.grid(True, linestyle='-', alpha=0.5, linewidth=1)

# 设置四周边框 (Box style)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

# 设置刻度向内看 (原图特征)
ax.tick_params(direction='in', top=True, right=True, width=1.5)

# 添加图例 (如果有必要)
# ax.legend(loc='upper right', frameon=True, edgecolor='black')

# 6. 调整布局并保存
plt.tight_layout()
# plt.savefig('step_plot.pdf', bbox_inches='tight') # 建议保存为PDF用于论文排版
plt.show()