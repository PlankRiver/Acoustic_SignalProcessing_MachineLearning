# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

PI = 3.1415927
h = 0.02
v = 1
tau = 0.2
A = v * h / tau
NX = 50
NT = 4000

Y = np.zeros((NX + 1, NT + 1), dtype=float)
Time = np.arange(0, NT + 1, 1)
X = np.arange(0, (NX + 1) * h, h)

# 初始条件
Y[:, 0] = np.sin(PI * X)  # y(x,t)=sin(pi*x)
# Y[:, 0] = np.sin(2 * PI * X)  # y(x,t)=sin(2*pi*x)
Y[:, 1] = Y[:, 0]  # dy/dt = 0

# 边界条件
Y[0, :] = 0.0
Y[NX, :] = 0.0

# 有限差分（空间向量化）
for k in range(1, NT):
    Y[1:NX, k + 1] = (
        2.0 * (1 - A**2) * Y[1:NX, k]
        + A**2 * (Y[2 : NX + 1, k] + Y[0 : NX - 1, k])
        - Y[1:NX, k - 1]
    )

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(X, Y[:, 0], "r-", label="t=0", linewidth=1.0)
ax1.plot(X, Y[:, 200], "b-", label="t=200", linewidth=1.0)
ax1.plot(X, Y[:, 300], "g-", label="t=300", linewidth=1.0)
ax1.plot(X, Y[:, 400], "k-", label="t=400", linewidth=1.0)
ax1.set_ylabel(r"Y", fontsize=20)
ax1.set_xlabel(r"X", fontsize=20)
ax1.set_xlim(0, 1)
ax1.set_ylim(-2, 2)
ax1.legend(loc="upper right", fontsize=10)

extent = [0, 250, 0, 50]
levels = np.arange(-2.0, 2.0, 0.01)
ax2.contourf(Y, levels, origin="lower", extent=extent, cmap=plt.cm.hot)
ax2.set_ylabel(r"X", fontsize=15)
ax2.set_xlabel(r"Time", fontsize=15)
plt.show()
