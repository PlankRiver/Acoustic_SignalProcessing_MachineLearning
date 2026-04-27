# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

tau = 0.05  # 时间步长
h = 0.5  # 空间步长
lamda = 1.0
A = 1.0 * tau / h**2
l = 10.0  # 开长
NX = int(l / h)  # 空间节点数
NY = int(l / h)  # 空间节点数
NT = 1000  # 时间步数

U = np.zeros((NX + 1, NY + 1, NT + 1), dtype=float)  # 温度
t = np.arange(0, NT, 1)  # 时间
x = np.arange(0, (NX + 1) * h, h)  # 坐标
y = np.arange(0, (NY + 1) * h, h)  # 坐标

# Time-invariant source on interior grid
Xc, Yc = np.meshgrid(x[1:NX], y[1:NY], indexing="ij")
source = tau * 5.0 * np.exp(-2.0 * ((Xc - 5.0) ** 2 + (Yc - 5.0) ** 2))

for k in range(NT):
    prev = U[:, :, k]
    nxt = U[:, :, k + 1]

    nxt[1:NX, 1:NY] = (
        (1 - 4.0 * A) * prev[1:NX, 1:NY]
        + A
        * (
            prev[2 : NX + 1, 1:NY]
            + prev[0 : NX - 1, 1:NY]
            + prev[1:NX, 2 : NY + 1]
            + prev[1:NX, 0 : NY - 1]
        )
        + source
    )

    # 第二类边界条件 (y-direction Neumann)
    nxt[:, 0] = nxt[:, 1]
    nxt[:, NY] = nxt[:, NY - 1]

    # 第一类边界条件 (x-direction Dirichlet)
    nxt[0, :] = 0.0
    nxt[NX, :] = 0.0

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

extent = [0, (NX + 1) * h, 0, (NY + 1) * h]
levels = np.arange(0.0, 3.5, 0.01)
ax1.contourf(U[:, :, 10].T, levels, origin="lower", extent=extent, cmap=plt.cm.hot, label="NT=10")
ax2.contourf(U[:, :, 100].T, levels, origin="lower", extent=extent, cmap=plt.cm.hot)
ax3.contourf(U[:, :, 300].T, levels, origin="lower", extent=extent, cmap=plt.cm.hot)
ax4.contourf(U[:, :, 1000].T, levels, origin="lower", extent=extent, cmap=plt.cm.hot)

ax1.set_ylabel("rY", fontsize=15)
ax3.set_xlabel("rX", fontsize=15)
ax3.set_ylabel("rY", fontsize=15)
ax4.set_xlabel("rX", fontsize=15)
ax1.set_title("NT=10")
ax2.set_title("NT=100")
ax3.set_title("NT=300")
ax4.set_title("NT=1000")
plt.show()
