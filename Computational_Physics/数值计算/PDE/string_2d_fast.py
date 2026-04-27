# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

PI = 3.1415927
h = PI / 70
v = 1
tau = 0.2
A = v * h / tau
NX = 70
NY = 70
NT = 250
target_k = 200

x = np.arange(0, NX + 1, 1)
y = np.arange(0, NY + 1, 1)

# 用滚动二维数组替代三维历史，显著减少内存与复制
u_prev = np.zeros((NX + 1, NY + 1), dtype=float)
u_curr = np.zeros((NX + 1, NY + 1), dtype=float)

Xg, Yg = np.meshgrid(x, y, indexing="ij")
u0 = 3 * np.sin(Xg * h) * np.sin(Yg * h)
u_prev[:, :] = u0
u_curr[:, :] = u0

# 边界条件
u_prev[0, :] = 0.0
u_prev[NX, :] = 0.0
u_prev[:, 0] = 0.0
u_prev[:, NY] = 0.0
u_curr[0, :] = 0.0
u_curr[NX, :] = 0.0
u_curr[:, 0] = 0.0
u_curr[:, NY] = 0.0

u_target = u_curr.copy() if target_k <= 1 else None

for k in range(1, NT):
    u_next = np.zeros_like(u_curr)
    u_next[1:NX, 1:NY] = (
        2 * u_curr[1:NX, 1:NY]
        - u_prev[1:NX, 1:NY]
        + A**2
        * (
            u_curr[2 : NX + 1, 1:NY]
            + u_curr[0 : NX - 1, 1:NY]
            - 4 * u_curr[1:NX, 1:NY]
            + u_curr[1:NX, 2 : NY + 1]
            + u_curr[1:NX, 0 : NY - 1]
        )
    )

    if k + 1 == target_k:
        u_target = u_next.copy()

    u_prev, u_curr = u_curr, u_next

if u_target is None:
    u_target = u_curr

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
Xm, Ym = np.meshgrid(x, y)
ax1.plot_wireframe(Xm, Ym, u_target, color="r")
ax1.set_ylabel(r"Y", fontsize=20)
ax1.set_xlabel(r"X", fontsize=20)
ax1.set_zlabel(r"U(X,Y)", fontsize=20)
plt.show()
