import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 设置全局学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# 2. 模拟数据准备 (模仿图中的多组多分类结构)
data_dict = {
    'Condition': ['WT']*20 + ['TDP-43-/-']*20,
    'Treatment': (['No treatment']*5 + ['TDP-43']*5 + ['TDP-43 ΔRRM']*5 + ['TDP-43 ΔIDR']*5) * 2,
    # 模拟均值和噪声
    'Value': np.concatenate([
        np.random.normal(1.5, 0.3, 5), np.random.normal(0.2, 0.1, 5), np.random.normal(2.5, 0.4, 5), np.random.normal(0.5, 0.2, 5),
        np.random.normal(7.0, 0.8, 5), np.random.normal(0.5, 0.1, 5), np.random.normal(15.0, 1.5, 5), np.random.normal(4.5, 0.6, 5)
    ])
}
df = pd.DataFrame(data_dict)

# 3. 设置配色 (高度还原原图配色)
colors = ["#a9c4a4", "#e7c84c", "#5a9bb6", "#9496c1"]

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# --- A. 绘制柱状图 (均值) ---
sns.barplot(
    data=df, x='Condition', y='Value', hue='Treatment',
    palette=colors, edgecolor='black', linewidth=1.5,
    errorbar=None,  # 稍后手动添加红色误差棒
    ax=ax, capsize=.1, alpha=0.9
)

# --- B. 绘制原始数据点 (抖动散点) ---
# 使用 stripplot，dodge=True 确保点对齐对应的柱子
sns.stripplot(
    data=df, x='Condition', y='Value', hue='Treatment',
    palette=['black']*4, dodge=True, size=6,
    jitter=0.15, alpha=0.8, ax=ax, legend=False
)

# --- C. 手动添加红色误差棒 (SEM) ---
# 获取柱子的位置信息
for i, patch in enumerate(ax.patches):
    # 计算均值和SEM
    # 注意：patch的顺序取决于分类，此处逻辑为示例简化版
    # 实际套用时可以直接利用 seaborn 的 errorbar 参数，但此处为还原红色，建议手动计算或后续染色
    x_val = patch.get_x() + patch.get_width() / 2
    y_val = patch.get_height()
    # 模拟SEM长度
    error = y_val * 0.15
    ax.errorbar(x_val, y_val, yerr=error, fmt='none', ecolor='red', elinewidth=2, capsize=5)

# 5. 添加显著性连线 (关键提分项)
def add_p_value(ax, x1, x2, y, h, p_text):
    """
    x1, x2: 横向索引
    y: 连线高度
    h: 钩子高度
    p_text: p值内容
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    ax.text((x1+x2)*.5, y+h, p_text, ha='center', va='bottom', color='black', fontsize=10)

# 示例：在大类 TDP-43-/- 内部添加 p 值
# 计算位置：Condition 1 的子组偏移
add_p_value(ax, 0.7, 0.9, 10, 0.5, "$p<0.0001$")
add_p_value(ax, 0.7, 1.3, 13, 0.5, "$p=0.1325$")

# 6. 图表美化
ax.set_title('Advanced Scientific Bar Plot', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Target Signal (RT-qPCR)', fontsize=13)
ax.set_xlabel('')

# 移除上方和右侧边框
sns.despine()

# 精修图例
ax.legend(title='', frameon=False, fontsize=11, loc='upper left')

# 坐标轴加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
# plt.savefig('nature_style_bar.pdf', bbox_inches='tight')
plt.show()