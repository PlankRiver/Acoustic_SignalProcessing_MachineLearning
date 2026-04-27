import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation

# 1. 设置全局学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.2)

# 2. 准备模拟数据
# 模拟不同类别和数值 (1e2 到 1e6 跨度)
categories = ['Lung', 'Breast', 'Prostate', 'Kidney', 'Liver', 'Skin', 'Uterus', 'Other']
colors = sns.color_palette("husl", len(categories))  # 鲜艳且易区分的配色


def generate_mock_data(n=500):
    cat_list = np.random.choice(categories, n)
    x = 10 ** np.random.normal(4, 0.7, n)
    y = x * np.random.normal(1, 0.5, n)  # 强相关
    return pd.DataFrame({'X': x, 'Y': y, 'Type': cat_list})


df = generate_mock_data(800)

# 3. 创建多面板画布 (以 c, d, e 三个小图为例)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
panel_labels = ['c', 'd', 'e']

# 4. 循环绘图 (模拟 c, d, e 风格)
for i, ax in enumerate(axes):
    # 绘制散点图
    sns.scatterplot(
        data=df, x='X', y='Y', hue='Type',
        palette=colors, s=40, alpha=0.8, edgecolor='none', ax=ax, legend=False
    )

    # 设置为对数坐标轴 (关键！)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 设置网格线 (主次网格都要有)
    ax.grid(True, which='both', linestyle='-', color='#e0e0e0', alpha=0.6)

    # 设置科学计数法刻度
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())

    # 添加阈值辅助线 (Dashed lines)
    x_threshold = 3e4
    y_threshold = 1.5e4
    ax.axvline(x_threshold, color='black', linestyle='--', linewidth=2)
    ax.axhline(y_threshold, color='black', linestyle='--', linewidth=2)

    # 添加辅助线旁边的文字标注
    ax.text(x_threshold * 1.1, 1e2, 'Threshold X', fontsize=10, fontweight='bold')
    ax.text(1.1e2, y_threshold * 1.1, 'Threshold Y', fontsize=10, fontweight='bold')

    # 标注小图编号 (a, b, c...)
    ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='right')

    # 轴标签美化
    ax.set_xlabel('Feature X (SNVs)', fontsize=14)
    ax.set_ylabel('Feature Y (Indels)', fontsize=14)

    # 坐标轴加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# 5. 创建底部共享图例
# 手动提取图例信息
handles, labels = [], []
for cat, col in zip(categories, colors):
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10))
    labels.append(cat)

fig.legend(handles, labels, loc='lower center', ncol=len(categories),
           bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=12)

# 6. 调整布局并显示
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# plt.savefig('advanced_scatter_grid.pdf', bbox_inches='tight')
plt.show()