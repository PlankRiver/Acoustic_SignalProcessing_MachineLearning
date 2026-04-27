import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

# 1. 构造模拟数据
# 行：Patient (ICC1-7)，列：Clinical features
data = {
    'Grade Group': ['GG3', 'GG3', 'GG3', 'GG3', 'GG3', 'GG3', 'GG5'],
    'PSA': ['>10', '<10', '<10', '>10', '>10', '>10', '>10'],
    'ICC': ['Present', 'Present', 'Present', 'Present', 'Present', 'Present', 'Present'],
    'IDC': ['Present', 'Present', 'Absent', 'Present', 'Present', 'Present', 'Present'],
    'Stage': ['<pT3a', '<pT3b', '<pT3a', '<pT3a', '<pT3a', '<pT3a', '<pT3b']
}
df = pd.DataFrame(data, index=[f'ICC{i}' for i in range(1, 8)])

# 2. 定义颜色映射表 (高度还原原图配色)
color_dict = {
    # Grade Group (紫色系/蓝色系)
    'GG2': '#a020f0', 'GG3': '#8b00ff', 'GG5': '#1e90ff',
    # PSA (青色系)
    '<10': '#00bfff', '>10': '#00ced1',
    # Present/Absent (绿色系)
    'Present': '#00fa9a', 'Absent': '#00cd00',
    # Stage (嫩绿色/黄色系)
    '<pT3a': '#9acd32', '<pT3b': '#eeee00'
}

# 3. 绘图准备
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

nrows, ncols = df.shape
# 创建一个空的背景，通过画矩形的方式来实现每个格子的独立着色
for r in range(nrows):
    for c in range(ncols):
        val = df.iloc[r, c]
        color = color_dict.get(val, 'white')
        # 画矩形，加上白色边框 (linewidth和edgecolor)
        rect = plt.Rectangle((c, nrows - r - 1), 1, 1,
                            facecolor=color, edgecolor='white', linewidth=3)
        ax.add_patch(rect)

# 4. 设置坐标轴
ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)

# 设置 X 轴标签 (旋转 30-45 度)
ax.set_xticks(np.arange(ncols) + 0.5)
ax.set_xticklabels(df.columns, rotation=30, ha='right', fontsize=14)
ax.set_xlabel('Clinical', fontsize=16, labelpad=20)

# 设置 Y 轴标签
ax.set_yticks(np.arange(nrows) + 0.5)
ax.set_yticklabels(df.index[::-1], fontsize=12)
ax.set_ylabel('Patient', fontsize=16)

# 移除坐标轴边框
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False)

# 5. 构建右侧图例 (Legend)
# 这里手动创建一个图例列表，匹配原图的标签
legend_labels = [
    ('Grade Group 2', '#a020f0'), ('Grade Group 3', '#8b00ff'), ('Grade Group 5', '#1e90ff'),
    ('PSA < 10 ng/mL', '#00bfff'), ('PSA > 10 ng/mL', '#00ced1'),
    ('Present', '#00fa9a'), ('Absent', '#00cd00'),
    ('<pT3a', '#9acd32'), ('<pT3b', '#eeee00')
]

legend_elements = [Patch(facecolor=c, label=l) for l, c in legend_labels]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
          frameon=False, fontsize=12, handlelength=1, handleheight=1)

# 6. 调整布局并保存
plt.tight_layout()
# plt.savefig('clinical_metadata_plot.pdf', bbox_inches='tight')
plt.show()