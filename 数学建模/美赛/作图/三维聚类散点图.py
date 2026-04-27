import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置全局字体（学术风格）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'


# 2. 准备模拟数据 (实际套用时请读取你的 X, Y, Z 和 Label 数据)
def generate_mock_3d_data(n=20000):
    np.random.seed(42)
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.zeros(n)
    labels = []

    for i in range(n):
        # 模拟地面 (Low vegetation)
        if np.random.rand() > 0.4:
            z[i] = np.random.uniform(0, 1)
            labels.append('Low vegetation')
        # 模拟屋顶 (Roof - 橙色方块)
        elif (20 < x[i] < 40 and 20 < y[i] < 40) or (60 < x[i] < 85 and 50 < y[i] < 75):
            z[i] = np.random.uniform(15, 18)
            labels.append('Roof')
        # 模拟树木 (Tree - 灰色簇)
        elif np.random.rand() > 0.8:
            z[i] = np.random.uniform(5, 12)
            labels.append('Tree')
        # 模拟立面/建筑侧面 (Facade - 绿色柱状)
        elif 40 < x[i] < 45 and 20 < y[i] < 40:
            z[i] = np.random.uniform(0, 15)
            labels.append('Facade')
        else:
            z[i] = np.random.uniform(0, 2)
            labels.append('Shrub')

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': labels})


df = generate_mock_3d_data(30000)

# 3. 定义分类及其对应的配色 (深度还原原图色系)
category_colors = {
    'Powerline': '#8dd3c7',  # 青色
    'Low vegetation': '#ffffb3',  # 淡黄
    'Impervious surfaces': '#bebada',  # 淡紫
    'Car': '#fb8072',  # 珊瑚红
    'Fence/Hedge': '#80b1d3',  # 浅蓝
    'Roof': '#fdb462',  # 橙色
    'Facade': '#b3de69',  # 嫩绿
    'Shrub': '#fccde5',  # 粉色
    'Tree': '#d9d9d9'  # 灰色
}

# 4. 创建 3D 画布
fig = plt.figure(figsize=(14, 10), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 5. 遍历每个类别进行绘制
for label, color in category_colors.items():
    sub_df = df[df['label'] == label]
    if not sub_df.empty:
        ax.scatter(sub_df['x'], sub_df['y'], sub_df['z'],
                   color=color,
                   label=label,
                   s=25,  # 点的大小
                   edgecolor='none',  # 无边框使颜色更纯净
                   alpha=0.8,  # 透明度防止重叠太死板
                   marker='o')

# 6. 美化细节
# 隐藏坐标轴和背景网格，模仿原图的“悬浮地图”感
ax.set_axis_off()
# 设置初始观察视角 (elev: 俯仰角, azim: 方位角)
ax.view_init(elev=35, azim=-45)

# 7. 自定义图例 (左上角)
legend = ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.05, 0.95),
    frameon=True,
    edgecolor='black',
    fontsize=16,
    handletextpad=1.2,
    labelspacing=0.8
)
# 加粗图例文字
for text in legend.get_texts():
    text.set_fontweight('bold')

# 8. 调整布局并显示
plt.tight_layout()
# plt.savefig('3d_point_cloud_classification.png', transparent=True, bbox_inches='tight')
plt.show()