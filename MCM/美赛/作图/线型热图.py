import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# ----------------------
# 1. 生成模拟数据（替换成你的真实数据）
# ----------------------
# 样本名称
samples = [
    'Samp A', 'Samp B', 'Samp C', 'Samp D', 'Samp E', 'Samp F', 'Samp G',
    'Samp H', 'Samp I', 'Samp J', 'Samp K', 'Samp L', 'Samp M', 'Samp N', 'Samp O'
]
n_samples = len(samples)
n_k = 90  # K轴的点数

# 生成模拟的线型热点数据
np.random.seed(42)
data = np.zeros((n_samples, n_k))
for i in range(n_samples):
    # 随机偏移热点位置
    center = np.random.randint(30, 50)
    # 生成钟形曲线模拟热点
    data[i] = 45 * np.exp(-((np.arange(n_k) - center) / 5)**2)
    # 加入随机噪声模拟平滑过渡
    data[i] += np.random.normal(0, 1, n_k) * 2

# ----------------------
# 2. 绘图设置
# ----------------------
fig, ax = plt.subplots(figsize=(14, 10))

# 使用与原图一致的配色（从深蓝到红）
cmap = cm.get_cmap('coolwarm')

# ----------------------
# 3. 绘制线型热图
# ----------------------
# 使用pcolormesh绘制带单元格分隔的热图
im = ax.pcolormesh(
    data, cmap=cmap, edgecolors='white', linewidth=0.3,
    vmin=0, vmax=45, shading='auto'
)

# ----------------------
# 4. 美化图表（美赛必备）
# ----------------------
# 设置坐标轴
ax.set_xticks(np.arange(0, n_k+1, 10))
ax.set_yticks(np.arange(n_samples) + 0.5)
ax.set_yticklabels(samples, fontsize=10)
ax.set_xlabel('K (w)', fontsize=12, fontweight='bold')
ax.set_ylabel('Samples', fontsize=12, fontweight='bold')

# 添加标题
ax.set_title('Linear Heatmap Plot', fontsize=16, fontweight='bold', pad=20)

# 添加颜色条
cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=15, pad=0.05)
cbar.set_label('Value', fontsize=10, fontweight='bold')
cbar.set_ticks(np.arange(0, 50, 5))

# 优化布局
plt.tight_layout()

# ----------------------
# 5. 仅显示图表（不保存PDF）
# ----------------------
plt.show()