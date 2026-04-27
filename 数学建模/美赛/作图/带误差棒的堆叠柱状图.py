import matplotlib.pyplot as plt
import numpy as np

# 1. 设置中文字体（如果是中文论文）和全局样式
# 美赛建议使用 Times New Roman。如果是中文环境，需要指定中文字体
plt.rcParams['font.sans-serif'] = ['SimSun'] # 宋体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'

# 2. 准备数据
labels = ['value1', 'value2', 'value3', 'value4', 'value5']

# 各个组成部分的高度 (近似原图数据)
dfj_data = np.array([1.5, 5.0, 10.0, 5.0, 1.5])
mfe_data = np.array([2.0, 1.2, 3.0, 1.2, 2.0])
zre_data = np.array([0.8, 1.2, 1.0, 1.2, 0.8])

# 各个部分的误差值 (用于画误差棒)
dfj_err = np.array([1.0, 1.5, 1.0, 1.5, 1.0])
mfe_err = np.array([0.5, 0.3, 1.0, 0.3, 0.5])
zre_err = np.array([0.8, 0.8, 1.0, 0.8, 0.8])

# 3. 颜色方案 (淡黄、浅绿)
colors = ['#ffec8b', '#c1e1c1', '#fce68d'] # dfj, mfe, zre

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 计算堆叠位置 (bottom参数)
bottom_mfe = dfj_data
bottom_zre = dfj_data + mfe_data

# 绘图逻辑：依次绘制每一层，并加上 yerr (误差棒)
# capsize 设置误差棒顶端横线宽度，ecolor 设置颜色
bar_width = 0.5

# 底层 dfj
ax.bar(labels, dfj_data, bar_width, label='dfj', color=colors[0],
       edgecolor='black', linewidth=0.8, yerr=dfj_err, capsize=15, ecolor='black')

# 中层 mfe
ax.bar(labels, mfe_data, bar_width, bottom=bottom_mfe, label='mfe', color=colors[1],
       edgecolor='black', linewidth=0.8, yerr=mfe_err, capsize=10, ecolor='black')

# 顶层 zre
ax.bar(labels, zre_data, bar_width, bottom=bottom_zre, label='zre', color=colors[2],
       edgecolor='black', linewidth=0.8, yerr=zre_err, capsize=12, ecolor='black')

# 5. 美化细节
ax.set_title('堆叠柱状图', fontsize=18, fontweight='bold', pad=20)
ax.set_ylabel('实验值', fontsize=14)
ax.set_xlabel('', fontsize=14)

# 坐标轴限制和刻度
ax.set_ylim(0, 16)
ax.set_yticks(np.arange(0, 17, 2))

# 只保留左侧和下方边框 (模仿原图风格)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 6. 图例设置 (调整顺序，让图例顺序与堆叠顺序一致)
handles, labels_legend = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels_legend[::-1], loc='upper right',
          frameon=False, fontsize=12, bbox_to_anchor=(0.95, 0.95))

# 7. 自动调整布局
plt.tight_layout()
# plt.savefig('stacked_bar_with_error.pdf', bbox_inches='tight') # 建议存为PDF
plt.show()