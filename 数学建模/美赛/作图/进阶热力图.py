import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据
# 横轴标签 (示例为基因/特征)
cols = ['NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'DLL1', 'DLL4', 'JAG1', 'JAG2']
# 纵轴标签 (示例为细胞状态/样本)
rows = ['SMC', 'EC_ven', 'EC_art', 'PC', 'EC_cap', 'Mesen', 'EC', 'EC_ln', 'PC_str', 'FB-like']

# 生成 0 到 1 之间的模拟数据
np.random.seed(42)
data = np.random.rand(len(rows), len(cols))

# 转换为 DataFrame 方便 Seaborn 处理
df = pd.DataFrame(data, index=rows, columns=cols)

# 2. 设置绘图风格
plt.rcParams['font.family'] = 'serif' # 美赛建议使用衬线字体
fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

# 3. 绘制热力图
# cmap: 选用 'RdPu' (红紫) 或 'BuPu'，非常接近原图的学术配色
# linewidths: 设置格子间的线条宽度
# linecolor: 设置线条颜色（浅灰色增加精致感）
# cbar_kws: 设置颜色条的标签
sns.heatmap(df,
            cmap='RdPu',
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='gray',
            cbar=True,
            ax=ax)

# 4. 美化坐标轴
# 设置 Y 轴标签旋转
ax.set_ylabel('cell_states', fontsize=16, labelpad=10)
plt.yticks(rotation=0, fontsize=12)

# 设置 X 轴标签旋转 (90度垂直)
plt.xticks(rotation=90, fontsize=12)

# 给热力图加一个黑色的外外框
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1)

# 5. 调整布局并保存
plt.tight_layout()
# plt.savefig('advanced_heatmap.pdf', bbox_inches='tight') # 建议保存为PDF
plt.show()