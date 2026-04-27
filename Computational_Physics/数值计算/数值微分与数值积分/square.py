import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.special import roots_legendre

# 数值计算部分
N = 101
l_half = 1.0
h = 2.0*l_half/(N-1)
k = 1.0
dx, dy = 0.1, 0.1
eps = 1e-8

xx = np.linspace(-l_half, l_half, N)

def lagendre_integratation(a,b,f,n):
    x,w = roots_legendre(n)
    x = 0.5*(b-a)*x+0.5*(b+a)
    w = 0.5*(b-a)*w
    s = 0.0
    for i in range(n):
        s += w[i]*f(x[i])
    return s



def U(x0,y0):
    top = lagendre_integratation(-l_half, l_half, lambda s: k * np.exp(-s**2) / np.sqrt((s - x0) ** 2 + (l_half - y0) ** 2 + eps), N//2)
    bottom = lagendre_integratation(-l_half, l_half, lambda s: k * np.exp(-s**2) / np.sqrt((s - x0) ** 2 + (-l_half - y0) ** 2 + eps), N//2)
    right = lagendre_integratation(-l_half, l_half, lambda s: k * np.exp(-s**2) / np.sqrt((l_half - x0) ** 2 + (s - y0) ** 2 + eps), N//2)
    left = lagendre_integratation(-l_half, l_half, lambda s: k * np.exp(-s**2) / np.sqrt((-l_half - x0) ** 2 + (s - y0) ** 2 + eps), N//2)
    return top + bottom + right + left


x_axis = np.arange(-3.0, 3.0, dx)
y_axis = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x_axis, y_axis)

potential = U(X, Y)
Ex = -(U(X+dx, Y) - U(X-dx, Y)) / (2*dx)
Ey = -(U(X, Y+dy) - U(X, Y-dy)) / (2*dy)
field = np.hypot(Ex, Ey)

# 绘图部分
vmax = np.percentile(np.abs(potential), 98)
vmax = max(vmax, 1.0)
levels = np.linspace(-vmax, vmax, 41)

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(13, 6), constrained_layout=True)


ax1 = fig.add_subplot(1, 2, 1)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cf = ax1.contourf(X, Y, potential, levels=levels, cmap='RdBu_r', norm=norm, alpha=0.96)
cs = ax1.contour(X, Y, potential, levels=np.linspace(-vmax, vmax, 13), colors='k', linewidths=0.6, alpha=0.55)
ax1.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
cb1 = fig.colorbar(cf, ax=ax1, shrink=0.9, pad=0.02)
cb1.set_label('Electric Potential (a.u.)', fontsize=11)


ax1.plot([-l_half, l_half, l_half, -l_half, -l_half], [l_half, l_half, -l_half, -l_half, l_half], color='#1f1f1f', linewidth=2.4, alpha=0.9, label='Charged square frame')
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

ax2.plot([-l_half, l_half, l_half, -l_half, -l_half], [l_half, l_half, -l_half, -l_half, l_half], color='white', linewidth=1.8, alpha=0.9)


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
ax1.set_title('Potential of Charged Square Frame', size=16, pad=10)
ax2.set_title('Electric Field Streamlines', size=16, pad=10)
ax1.tick_params(labelsize=11)
ax2.tick_params(labelsize=11)

plt.show()