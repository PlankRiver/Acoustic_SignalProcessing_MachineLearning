import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 1. 定义兰彻斯特方程 (平方律)
def lanchester_square(state, t, alpha, beta):
    x, y = state  # x: 红军, y: 蓝军

    # 简单的边界处理：如果人数小于0，不再减少（或者设为0）
    if x <= 0: x = 0
    if y <= 0: y = 0

    dxdt = -beta * y  # 红军死的速率 = 蓝军人数 * 蓝军效率
    dydt = -alpha * x  # 蓝军死的速率 = 红军人数 * 红军效率

    # 防止死成负数
    if x <= 0: dxdt = 0
    if y <= 0: dydt = 0

    return [dxdt, dydt]


# 2. 设置参数
x0 = 1000  # 红军人数
y0 = 500  # 蓝军人数
alpha = 0.1  # 红军单兵效率
beta = 0.5  # 蓝军单兵效率

t = np.linspace(0, 10, 1000)  # 时间轴

# 3. 求解
solution = odeint(lanchester_square, [x0, y0], t, args=(alpha, beta))
red_troops = solution[:, 0]
blue_troops = solution[:, 1]

# 4. 可视化
plt.figure(figsize=(10, 6))
plt.plot(t, red_troops, 'r-', label='Red Army (More People, Low Tech)')
plt.plot(t, blue_troops, 'b-', label='Blue Army (Elite, High Tech)')

# 找到一方全军覆没的时间点
winner_idx = np.where(red_troops <= 0.1)[0]
if len(winner_idx) > 0:
    end_time = t[winner_idx[0]]
    plt.axvline(x=end_time, color='k', linestyle='--', label=f'Red Eliminated at t={end_time:.2f}')
else:
    winner_idx = np.where(blue_troops <= 0.1)[0]
    if len(winner_idx) > 0:
        end_time = t[winner_idx[0]]
        plt.axvline(x=end_time, color='k', linestyle='--', label=f'Blue Eliminated at t={end_time:.2f}')

plt.title(f'Lanchester Square Law Model\nInitial: Red={x0}(eff={alpha}), Blue={y0}(eff={beta})')
plt.xlabel('Time')
plt.ylabel('Troops Remaining')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()