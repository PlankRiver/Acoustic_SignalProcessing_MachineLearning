# 0、一共有n辆车
# 1、汽车当前速度为v
# 2、如果前方无车则下一时刻速度提升为v+1，直到最高速度
# 3、如果前方有车，距离d<v，则下一时刻速度减小为d-1
# 4、每辆车会以概率p随即减速v-1

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# 参数设置
path = 5000  # 环形公路的长度
n = 100  # 公路中的车辆数目
v0 = 5   #车辆的初始速度
p = 0.6  # 随机减速概率
times = 3000  # 模拟时间步长

np.random.seed(0)
x = np.random.rand(n) * path  # 随机初始化车辆位置
x.sort()  # 按位置排序
v = np.full(n, v0, dtype=float)  # 初始化速度

# 直接绘制所有点数据，而不是逐步绘制
points_data = []

for t in range(times):
    if t % 5 == 0:
        for xi in x:
            points_data.append([xi, t])

    # 每辆车与前车的距离
    d = np.roll(x, -1) - x
    d[d <= 0] += path
    mask = v < d  # 判断是否有足够空间加速
    # 可以加速
    v[mask] += 1
    # 随机减速
    random_slow = np.random.rand(n) < p
    v[random_slow] -= 1
    # 需要减速
    v[~mask] = d[~mask] - 1

    # 更新位置
    x += v
    x %= path

# 转换为numpy数组并绘制
points_data = np.array(points_data)

fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
ax.scatter(points_data[:, 0], points_data[:, 1], s=1, c='k', alpha=0.5)

ax.set_xlim(0, path)
ax.set_ylim(0, times)
ax.set_xlabel('Position', fontsize=16)
ax.set_ylabel('Time', fontsize=16)
ax.set_title('Traffic Simulation - Car Jam Model', fontsize=18)
plt.tight_layout(pad=2)
plt.savefig('traffic_simulation.png', dpi=100, bbox_inches='tight')
print("图表已保存为 traffic_simulation.png")
plt.show()