import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 设置全局学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 准备数据
# 模拟一个 20x20 的矩阵
np.random.seed(42)
data = np.random.randint(20, 70, size=(20, 20))

# 技巧：将对数角线设为 NaN，使其在图中显示为白色 (模仿原图特征)
mask_diag = np.eye(data.shape[0], dtype=bool)
plot_data = data.astype(float)
plot_data[mask_diag] = np.nan

# 转换为 DataFrame 方便打标签
df = pd.DataFrame(plot_data)

# 3. 创建画布
fig, ax = plt.subplots(figsize=(12, 11), dpi=150)

# 4. 绘制热力图
# annot: 是否在格子里显示数字
# fmt: 数字格式，'d'为整数，'.1f'为一位小数
# cmap: 选用 'YlGnBu' (黄-绿-蓝) 非常接近原图，或者 'viridis'
# linewidths: 格子之间的白线宽度 (原图灵魂)
# cbar: 是否显示右侧颜色条
sns.heatmap(df,
            annot=True,
            fmt='.0f',
            cmap='YlGnBu',
            linewidths=0.5,
            linecolor='white',
            annot_kws={"size": 10}, # 调整格子内字体大小
            cbar=True,
            ax=ax)

# 5. 美化细节
ax.set_title('Standard Heatmap Template', fontsize=18, fontweight='bold', pad=20)

# 如果你的数据有特定的含义，可以在这里设置标签
# ax.set_xlabel('Parameter X', fontsize=14)
# ax.set_ylabel('Parameter Y', fontsize=14)

# 保持热力图为正方形
ax.set_aspect('equal')

# 设置刻度标签
# plt.xticks(np.arange(20)+0.5, [f'V{i}' for i in range(20)])
# plt.yticks(np.arange(20)+0.5, [f'V{i}' for i in range(20)])

# 6. 调整布局并保存
plt.tight_layout()
# plt.savefig('standard_heatmap.pdf', bbox_inches='tight')
plt.show()