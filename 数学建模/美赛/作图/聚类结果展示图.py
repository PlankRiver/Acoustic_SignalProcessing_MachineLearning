import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 设置全局样式：加粗字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'


# 2. 准备模拟数据 (实际套用时请读取你的数据：X坐标, Y坐标, 分类标签)
def generate_mock_data(n_points=30000):
    np.random.seed(42)
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 60, n_points)

    # 简单的逻辑来模拟图中的“块状”分布
    labels = []
    for xi, yi in zip(x, y):
        if 20 < xi < 45 and 35 < yi < 55:
            labels.append('Roof')
        elif 55 < xi < 85 and 30 < yi < 50:
            labels.append('Roof')
        elif 10 < xi < 95 and 25 < yi < 38:
            labels.append('Impervious surfaces')
        elif yi < 20:
            labels.append('Low vegetation')
        elif 50 < xi < 58 and yi > 40:
            labels.append('Facade')
        elif 10 < xi < 15 and 35 < yi < 58:
            labels.append('Fence/Hedge')
        elif np.random.rand() > 0.85:
            labels.append('Tree')
        elif np.random.rand() > 0.7:
            labels.append('Shrub')
        else:
            labels.append('Low vegetation')
    return pd.DataFrame({'x': x, 'y': y, 'label': labels})


df = generate_mock_data(50000)

# 3. 定义分类及其对应的配色 (完全还原原图配色)
category_colors = {
    'Powerline': '#8dd3c7',  # 青绿色
    'Low vegetation': '#ffffb3',  # 淡黄色
    'Impervious surfaces': '#bebada',  # 淡紫色
    'Car': '#fb8072',  # 珊瑚红
    'Fence/Hedge': '#80b1d3',  # 浅蓝色
    'Roof': '#fdb462',  # 橙黄色
    'Facade': '#b3de69',  # 嫩绿色
    'Shrub': '#fccde5',  # 粉红色
    'Tree': '#d9d9d9'  # 灰色
}

# 4. 绘图
fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

# 遍历每个类别进行绘制
# 点的大小 s 需要设置得足够大，使其产生“面”的视觉效果
for label, color in category_colors.items():
    sub_df = df[df['label'] == label]
    ax.scatter(sub_df['x'], sub_df['y'],
               color=color,
               label=label,
               s=55,  # 点的大小，根据数据密度调整
               edgecolor='none',
               marker='o')

# 5. 美化细节：完全隐藏坐标轴和边框
ax.set_axis_off()

# 6. 自定义图例 (关键：位置、大小、加粗)
legend = ax.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),  # 将图例放在主图右侧
    frameon=False,  # 去掉图例边框
    fontsize=16,  # 字体调大
    handletextpad=1.5,  # 图标和文字的间距
    labelspacing=1.2  # 各行图例的间距
)

# 强制将图例文字加粗
for text in legend.get_texts():
    text.set_fontweight('bold')

# 7. 调整布局并显示
plt.tight_layout()
# 如果需要保存图片，取消下面一行的注释
# plt.savefig('spatial_map.png', transparent=True, bbox_inches='tight')
plt.show()