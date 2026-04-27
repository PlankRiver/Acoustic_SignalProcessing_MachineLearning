import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import roots_legendre

# 数值计算部分
N = 1000
cycle = 2.0 * np.pi
d_theta = cycle / N
Q = 1.0
R = 1.0
dx, dy = 0.1, 0.1


theta = np.linspace(0, cycle, N, endpoint=False)

def f(theta,x0,y0):
    return Q / (2.0 * np.pi) * R * d_theta / np.sqrt((R * np.cos(theta) - x0) ** 2 + (R * np.sin(theta) - y0) ** 2)

def lagendre_integratation(a,b,f,n):
    x,w = roots_legendre(n)
    x = 0.5*(b-a)*x+0.5*(b+a)
    w = 0.5*(b-a)*w
    s = 0.0
    for i in range(n):
        s += w[i]*f(x[i])
    return s


def U(x0,y0):
    return lagendre_integratation(0, cycle, lambda theta: f(theta, x0, y0), N//2)


x_axis = np.arange(-3.0, 3.0, dx)
y_axis = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x_axis, y_axis)

potential = U(X, Y)
Ex = -(U(X+dx, Y) - U(X-dx, Y)) / (2*dx)
Ey = -(U(X, Y+dy) - U(X, Y-dy)) / (2*dy)
field = np.hypot(Ex, Ey)


# 绘图部分
vmin = max(np.percentile(potential, 2), 1e-12)
vmax = np.percentile(potential, 99.7)
levels = np.geomspace(vmin, vmax, 60)

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 6), constrained_layout=True)


ax1 = fig.add_subplot(1, 2, 1)
norm = LogNorm(vmin=vmin, vmax=vmax)
cf = ax1.contourf(X, Y, potential, levels=levels, cmap='inferno', norm=norm, alpha=0.98)
cs = ax1.contour(X, Y, potential, levels=np.geomspace(vmin, vmax, 12), colors='white', linewidths=0.7, alpha=0.65)
ax1.clabel(cs, inline=True, fontsize=8, fmt='%.1e')
cb1 = fig.colorbar(cf, ax=ax1, shrink=0.9, pad=0.02)
cb1.set_label('Electric Potential (a.u.)', fontsize=11)

ax1.plot(R * np.cos(theta), R * np.sin(theta), color='#1f1f1f', linewidth=2.4, alpha=0.9, label='Charged segment')
ax1.legend(loc='upper right', fontsize=9, frameon=True)

ax2 = fig.add_subplot(1, 2, 2)
log_field = np.log10(field + 1e-10)
stream = ax2.streamplot(
    X, Y, Ex, Ey,
    color=log_field,
    density=1.7,
    linewidth=1.0,
    arrowsize=1.15,
    cmap='magma'
)
cb2 = fig.colorbar(stream.lines, ax=ax2, shrink=0.9, pad=0.02)
cb2.set_label('log10|E| (a.u.)', fontsize=11)

ax2.plot(R * np.cos(theta), R * np.sin(theta), color='white', linewidth=1.8, alpha=0.9)

ax1.set_xlabel('x', size=20)
ax1.set_ylabel('y', size=20)
ax2.set_xlabel('x', size=20)
ax2.set_ylabel('y', size=20)

ax1.set_xlim(-3.0, 3.0)
ax1.set_ylim(-3.0, 3.0)
ax2.set_xlim(-3.0, 3.0)
ax2.set_ylim(-3.0, 3.0)
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
ax1.set_title('Potential of Continuous Charge', size=16, pad=10)
ax2.set_title('Electric Field Streamlines', size=16, pad=10)
ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)

plt.show()