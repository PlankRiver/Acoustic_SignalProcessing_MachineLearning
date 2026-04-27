import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. 定义微分方程
def malthus_growth(P, t, r):
    """Malthus 模型: dP/dt = rP"""
    return r * P

def logistic_growth(P, t, r, K):
    """Logistic 模型: dP/dt = rP(1 - P/K)"""
    return r * P * (1 - P / K)

# 2. 设置参数
P0 = 10      # 初始人口
r = 0.1      # 增长率 (10%)
K = 1000     # 环境最大容纳量
t = np.linspace(0, 100, 200) # 时间跨度 0-100年

# 3. 求解微分方程 (odeint)
# args 必须是 tuple 形式
P_malthus = odeint(malthus_growth, P0, t, args=(r,))
P_logistic = odeint(logistic_growth, P0, t, args=(r, K))

# 4. 可视化对比
plt.figure(figsize=(10, 6))

# 绘制 Malthus (只画前一部分，否则会飞出天际)
plt.plot(t[:100], P_malthus[:100], 'r--', label='Malthus (Exponential) - J Curve')

# 绘制 Logistic
plt.plot(t, P_logistic, 'b-', linewidth=2, label='Logistic (Verhulst) - S Curve')

# 绘制环境上限 K
plt.axhline(y=K, color='g', linestyle=':', label=f'Carrying Capacity K={K}')

plt.title('Population Growth Models Comparison')
plt.xlabel('Time (t)')
plt.ylabel('Population (P)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()