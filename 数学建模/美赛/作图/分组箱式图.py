import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 设置全局样式（清爽的学术风格）
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 构造模拟数据 (实际套用时请读取你的 CSV 文件)
# 假设有 4 个大类 (axolotl, fugu, human, lungfish)
# 每个大类下有 5 个子组 (1st, 2nd, 3rd, 4th, 4th+)
groups = ['axolotl', 'fugu', 'human', 'lungfish']
sub_groups = ['1st', '2nd', '3rd', '4th', '4th+']

data_list = []
for g in groups:
    for sg in sub_groups:
        # 模拟生成一些对数分布的数据
        base_val = np.random.uniform(7, 15)
        size = 100
        values = np.random.normal(loc=base_val, scale=1.5, size=size)
        # 添加一些离群点
        outliers = np.random.uniform(5, 20, size=10)
        all_vals = np.concatenate([values, outliers])

        for v in all_vals:
            data_list.append({'Category': g, 'Stage': sg, 'Value': v})

df = pd.DataFrame(data_list)

# 3. 定义颜色（参考原图的鲜艳配色）
my_palette = ["#ff7f7f", "#999900", "#00b36b", "#1aa3ff", "#e64dff"]

# 4. 创建画布
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# 5. 绘制分组箱线图
sns.boxplot(
    data=df,
    x='Category',
    y='Value',
    hue='Stage',
    palette=my_palette,
    width=0.8,  # 箱子宽度
    linewidth=1.2,  # 箱子边框粗细
    fliersize=3,  # 离群点大小
    flierprops={"markerfacecolor": "0.2", "marker": "o", "markeredgecolor": "none"},  # 离群点样式
    medianprops={"color": "black", "linewidth": 2}  # 中位数线加粗
)

# 6. 图表精修
ax.set_ylabel('log2(bp)', fontsize=14, fontweight='bold')
ax.set_xlabel('', fontsize=14)  # 通常大类标签已在刻度上，不需要额外说明

# 移除上方和右侧的边框 (Spines)
sns.despine()

# 调整刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 7. 调整图例 (放在右侧中央)
plt.legend(title='', loc='center left', bbox_to_anchor=(1, 0.5),
           frameon=False, fontsize=12)

# 8. 布局并保存
plt.tight_layout()
# plt.savefig('grouped_boxplot.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()