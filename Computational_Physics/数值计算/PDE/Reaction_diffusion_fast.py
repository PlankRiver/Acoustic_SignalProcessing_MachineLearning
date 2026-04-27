# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

N = 50
Nhalf = N // 2
Nt = 3000000
tau = 1.0  # 时间步长
h = 1.0  # 空间步长
Du, Dv, f, r = 0.16, 0.08, 0.060, 0.062
# Du, Dv, f, r = 0.14, 0.06, 0.035, 0.065

Au = Du * tau / h**2
Av = Dv * tau / h**2

# Rolling 2D states (fast mode)
U = 0.02 * np.random.random((N + 1, N + 1))
V = 0.02 * np.random.random((N + 1, N + 1))
U[Nhalf - 16 : Nhalf + 16, Nhalf - 16 : Nhalf + 16] = 0.50
V[Nhalf - 16 : Nhalf + 16, Nhalf - 16 : Nhalf + 16] = 0.25

for _ in range(Nt):
    Un = U.copy()
    Vn = V.copy()

    u = U[1:N, 1:N]
    v = V[1:N, 1:N]
    uvv = u * v * v

    Un[1:N, 1:N] = (
        (1 - 4.0 * Au) * u
        + Au * (U[2 : N + 1, 1:N] + U[0 : N - 1, 1:N] + U[1:N, 2 : N + 1] + U[1:N, 0 : N - 1])
        - tau * uvv
        + tau * f * (1.0 - u)
    )

    Vn[1:N, 1:N] = (
        (1 - 4.0 * Av) * v
        + Av * (V[2 : N + 1, 1:N] + V[0 : N - 1, 1:N] + V[1:N, 2 : N + 1] + V[1:N, 0 : N - 1])
        + tau * uvv
        - tau * (f + r) * v
    )

    # Periodic-like boundaries (same rule as original script)
    Un[1:N, 0] = Un[1:N, N - 1]
    Un[1:N, N] = Un[1:N, 1]
    Vn[1:N, 0] = Vn[1:N, N - 1]
    Vn[1:N, N] = Vn[1:N, 1]

    Un[0, :] = Un[N - 1, :]
    Un[N, :] = Un[1, :]
    Vn[0, :] = Vn[N - 1, :]
    Vn[N, :] = Vn[1, :]

    U, V = Un, Vn

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.pcolor(U, cmap=plt.cm.RdBu)
ax2.pcolor(V, cmap=plt.cm.RdBu)
plt.show()
