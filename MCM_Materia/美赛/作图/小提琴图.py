import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------
# 1. 生成模拟数据（替换成你的真实数据）
# ----------------------
np.random.seed(42)

# 5个样本的数据
sample1 = np.random.normal(loc=1, scale=1.5, size=200)
sample2 = np.random.normal(loc=0.5, scale=0.8, size=200)
sample3 = np.random.normal(loc=1.5, scale=2.0, size=200)
sample4 = np.random.normal(loc=2.0, scale=1.2, size=200)
sample5 = np.random.normal(loc=0.8, scale=1.4, size=200)

data = [sample1, sample2, sample3, sample4, sample5]
labels = ['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5']

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(12, 10))

# 配色（和原图一致）
colors = ['#8ECAE6', '#FFB703', '#C8B6A6', '#FB8500', '#219EBC']

# ----------------------
# 3. 绘制小提琴图
# ----------------------
# 绘制小提琴主体
violins = ax.violinplot(
    data,
    showmeans=False,
    showmedians=True,
    showextrema=True
)

# 为每个小提琴设置颜色
for i, pc in enumerate(violins['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

# 设置中位数和极值线样式
violins['cmedians'].set_color('black')
violins['cmedians'].set_linewidth(2)
violins['cmaxes'].set_color('black')
violins['cmins'].set_color('black')
violins['cbars'].set_color('black')

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 设置坐标轴
ax.set_xticks(np.arange(1, len(labels)+1))
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Δ [yesno⁻²]', fontsize=14, fontweight='bold')
ax.set_ylim(-3, 6)

# 添加标题
ax.set_title('Violin plot', fontsize=18, fontweight='bold', pad=20)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3)

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()