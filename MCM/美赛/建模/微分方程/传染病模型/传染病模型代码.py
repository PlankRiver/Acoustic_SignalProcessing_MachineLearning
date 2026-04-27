import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# 1. 定义 SEIR 微分方程系统
def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]


# 2. 设置参数
N = 100000  # 总人口 10万
beta = 0.3  # 感染率 (每天传染 0.3 个人)
D_incubation = 5.0  # 潜伏期 5 天
D_infectious = 10.0  # 感染期 10 天

sigma = 1.0 / D_incubation
gamma = 1.0 / D_infectious

R0 = beta / gamma
print(f"当前模型的 R0 = {R0:.2f}")

# 初始状态
I0 = 10  # 初始感染者
E0 = 20  # 初始潜伏者
R0_state = 0  # 初始康复者
S0 = N - I0 - E0 - R0_state

t = np.linspace(0, 160, 160)  # 模拟 160 天

# 3. 求解
y0 = [S0, E0, I0, R0_state]
ret = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# 4. 可视化
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible (易感)')
plt.plot(t, E, 'y', alpha=0.5, lw=2, label='Exposed (潜伏)')
plt.plot(t, I, 'r', alpha=0.8, lw=3, label='Infected (确诊)')
plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered (康复)')

plt.title(f'SEIR Model Simulation (Population={N}, R0={R0:.1f})')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.legend()
plt.grid(True, alpha=0.3)

# 标记峰值
peak_idx = np.argmax(I)
plt.plot(t[peak_idx], I[peak_idx], 'ko')
plt.text(t[peak_idx], I[peak_idx] + 2000, f'Peak: {int(I[peak_idx])} people\nDay: {int(t[peak_idx])}', ha='center')

plt.show()