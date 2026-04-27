import matplotlib.pyplot as plt
import numpy as np


# 1. 模拟数据生成 (美赛中请替换为你处理后的数据)
def generate_mock_data(n=50):
    x = np.random.rand(n) * 50
    y = np.random.rand(n) * 50
    z = np.random.rand(n) * 50
    size = np.random.rand(n) * 500  # 气泡大小
    color = np.random.rand(n)  # 颜色映射值 (0-1)
    return x, y, z, size, color


# 2. 设置绘图风格 (美赛建议使用 'seaborn-v0_8-whitegrid' 或 'science' 风格)
plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure(figsize=(12, 10))
titles = ['Subplot A', 'Subplot B', 'Subplot C', 'Subplot D']

# 3. 循环创建 2x2 的子图
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    x, y, z, s, c = generate_mock_data()

    # 核心绘图函数 scatter
    # cmap='viridis' 对应你图中的绿黄紫色调
    img = ax.scatter(x, y, z, s=s, c=c, cmap='viridis', alpha=0.8, edgecolors='w', linewidth=0.5)

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(titles[i - 1], fontsize=12, fontweight='bold')

    # 调整视角 (根据数据特点调整 elev 和 azim)
    ax.view_init(elev=20, azim=45)

# 4. 添加公共颜色条 (Colorbar)
# 调整 location 和 shrink 参数使其美观
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [左, 下, 宽, 高]
fig.colorbar(img, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为颜色条留出空间
plt.show()