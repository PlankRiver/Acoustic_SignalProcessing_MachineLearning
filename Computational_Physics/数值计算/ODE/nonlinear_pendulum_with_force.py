import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

# How to solve d^2theta/dt^2 = -g/l*sin(theta)
# We can use RK4 with:
# dtheta/dt = omega    dx/dt = f(x,y,t)  x(0) = x0
# domega/dt = -g/l*sin(theta)-qdtheta/dt+Fsin(wt)    dy/dt = g(x,y,t)  y(0) = y0

# Chaotic System

g = 9.8
l = 10.0
q = 0.5
F = 1.5
w = 2.0/3.0
theta0 = 0.2
omega0 = 0.0
theta = theta0
omega = omega0
t = 0.0
dt = 0.01
steps = 3000

theta_RK4 = []; theta_RK4.append(theta)
omega_RK4 = []; omega_RK4.append(omega)
time = []; time.append(t)

def ftheta(theta,omega,t):
    return omega

def gomega(theta,omega,t):
    return -g/l*np.sin(theta)-q*omega+F*np.sin(w*t)

def RK4(theta,omega,t):
    k1 = ftheta(theta,omega,t)
    l1 = gomega(theta,omega,t)
    k2 = ftheta(theta+k1*dt/2,omega+l1*dt/2,t)
    l2 = gomega(theta+k1*dt/2,omega+l1*dt/2,t)
    k3 = ftheta(theta+k2*dt/2,omega+l2*dt/2,t)
    l3 = gomega(theta+k2*dt/2,omega+l2*dt/2,t)
    k4 = ftheta(theta+k3*dt,omega+l3*dt,t)
    l4 = gomega(theta+k3*dt,omega+l3*dt,t)
    k_avg = (k1+2*k2+2*k3+k4)/6
    l_avg = (l1+2*l2+2*l3+l4)/6
    theta += k_avg*dt
    omega += l_avg*dt
    if theta > 2*np.pi:
        theta -= 2*np.pi
    if theta < -2*np.pi:
        theta += 2*np.pi

    return theta,omega


for i in range(steps):
    theta,omega = RK4(theta,omega,t)
    t += dt

    theta_RK4.append(theta)
    omega_RK4.append(omega)
    time.append(t)

plt.plot(time, theta_RK4, 'r-', label='The evaluation of theta with time')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.title('Theta Evaluation')
plt.grid()
plt.show()

plt.plot(time, omega_RK4, 'r-', label='The evaluation of omega with time')
plt.xlabel('Time')
plt.ylabel('Omega')
plt.title('Omega Evaluation')
plt.grid()
plt.show()

plt.plot(theta_RK4, omega_RK4, 'r-', label='The phase graph')
plt.xlabel('Theta')
plt.ylabel('Omega')
plt.title('Phase Graph')
plt.grid()
plt.show()


# Animate the pendulum motion: draw the rod and bob at each time step.
x = l * np.sin(np.array(theta_RK4))
y = -l * np.cos(np.array(theta_RK4))
tail_length = 200
save_mp4 = True
mp4_name = 'nonlinear_pendulum_with_force.mp4'
output_path = Path(__file__).resolve().parent / mp4_name

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.2 * l, 1.2 * l)
ax.set_ylim(-1.2 * l, 0.2 * l)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Nonlinear Pendulum Animation')
ax.grid()

rod_line, = ax.plot([], [], 'b-', lw=2)
trail_line, = ax.plot([], [], 'g-', lw=1.5, alpha=0.7)
bob_point, = ax.plot([], [], 'ro', ms=10)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init_anim():
    rod_line.set_data([], [])
    trail_line.set_data([], [])
    bob_point.set_data([], [])
    time_text.set_text('')
    return rod_line, trail_line, bob_point, time_text

def update_anim(frame):
    x_bob = x[frame]
    y_bob = y[frame]
    start = max(0, frame - tail_length)
    rod_line.set_data([0.0, x_bob], [0.0, y_bob])
    trail_line.set_data(x[start:frame + 1], y[start:frame + 1])
    bob_point.set_data([x_bob], [y_bob])
    time_text.set_text(f't = {time[frame]:.2f} s')
    return rod_line, trail_line, bob_point, time_text

ani = FuncAnimation(
    fig,
    update_anim,
    frames=len(time),
    init_func=init_anim,
    interval=dt * 1000,
    blit=True,
    repeat=True,
)

if save_mp4:
    fps = max(1, int(round(1.0 / dt)))
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(str(output_path), writer=writer)
        print(f'MP4 saved to: {output_path}')
    except Exception as e:
        print('MP4 save failed. Install ffmpeg and ensure it is in PATH.')
        print(f'Error: {e}')

plt.show()