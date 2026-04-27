import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 设置全局样式（学术风格）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.2)

# 2. 模拟数据 (5个大类，每个大类下有2组数据)
categories = ['Nan', 'MT', 'Lowdo', 'Midean', 'Highest']
groups = ['Type A', 'Type B']

data = []
for cat in categories:
    for grp in groups:
        # 模拟正态分布数据，并添加一些异常值
        base = np.random.uniform(3.5, 5.0)
        vals = np.random.normal(loc=base, scale=0.5, size=50)
        # 模拟离群点 (outliers)
        outliers = np.random.uniform(1.0, 2.5, size=3)
        vals = np.concatenate([vals, outliers])
        for v in vals:
            data.append({'Category': cat, 'Group': grp, 'Value': v})

df = pd.DataFrame(data)

# 3. 绘图设置
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

# 定义颜色：淡蓝色和白色
palette = {'Type A': '#d1e5f0', 'Type B': '#ffffff'}

# 4. 绘制箱线图
# flierprops 用于设置异常值样式（红色+号）
boxplot = sns.boxplot(
    data=df, x='Category', y='Value', hue='Group',
    palette=palette, width=0.6, linewidth=1.2,
    fliersize=6, flierprops={'marker': '+', 'markeredgecolor': 'red'},
    whiskerprops={'linestyle': '--'}, # 须线设为虚线
    ax=ax
)

# 5. 添加显著性标记 (Significance Brackets)
# 这是一个手动添加连线的函数
def add_significance(ax, x1, x2, y, h, text):
    """
    x1, x2: 连线的两个横坐标索引
    y: 连线的高度
    h: 连线两端垂直下降的高度
    text: 显示的文字（如 '*'）
    """
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y, text, ha='center', va='bottom', color='black', fontsize=20, fontweight='bold')

# 根据原图位置添加星号 (假设 Lowdo 与 Highest 之间有显著差异)
# 计算位置：大类索引 + 组内偏移
add_significance(ax, 2.2, 4.2, 6.2, 0.2, "*")
add_significance(ax, 2.0, 4.0, 7.0, 0.5, "*")

# 6. 图表美化
ax.set_title('Grouped Boxplot with Significance', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Value (Kenan)', fontsize=14)
ax.set_xlabel('Category (花键)', fontsize=14)

# 设置刻度向内
ax.tick_params(direction='in', length=6, width=1.2, top=True, right=True)

# 移除图例（如果不需要的话）或者精修图例
ax.legend_.remove()

# 限制坐标轴范围
ax.set_ylim(1, 8)

# 加粗边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 7. 布局与保存
plt.tight_layout()
# plt.savefig('grouped_boxplot_significance.pdf', bbox_inches='tight')
plt.show()