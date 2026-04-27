import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


G = 10.0
SOFTENING = 1e-9


def acceleration(positions, masses):
	diff = positions[None, :, :] - positions[:, None, :]
	dist2 = np.sum(diff**2, axis=2) + SOFTENING**2
	np.fill_diagonal(dist2, np.inf)
	inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
	return G * np.sum(diff * masses[None, :, None] * inv_dist3[:, :, None], axis=1)


def derivative(state, masses):
	n = len(masses)
	positions = state[: 2 * n].reshape(n, 2)
	velocities = state[2 * n :].reshape(n, 2)
	acc = acceleration(positions, masses)
	return np.concatenate([velocities.ravel(), acc.ravel()])


def rk4_step(state, dt, masses):
	k1 = derivative(state, masses)
	k2 = derivative(state + 0.5 * dt * k1, masses)
	k3 = derivative(state + 0.5 * dt * k2, masses)
	k4 = derivative(state + dt * k3, masses)
	return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(positions0, velocities0, masses, dt, steps):
	positions0 = np.asarray(positions0, dtype=float)
	velocities0 = np.asarray(velocities0, dtype=float)
	masses = np.asarray(masses, dtype=float)
	n = len(masses)

	state = np.concatenate([positions0.ravel(), velocities0.ravel()])
	trajectory = np.empty((steps + 1, n, 2), dtype=float)
	trajectory[0] = positions0

	for i in range(steps):
		state = rk4_step(state, dt, masses)
		trajectory[i + 1] = state[: 2 * n].reshape(n, 2)

	times = np.linspace(0.0, steps * dt, steps + 1)
	return times, trajectory


def chaotic_initial_conditions():
	# 改这里：三体初始位置（每行一个质点的 x, y）
	positions0 = np.array(
		[
			[-10.0, 0.0],
			[8.35, 10.9],
			[7.8, -8.75],
		]
	)
	# 改这里：三体初始速度（每行一个质点的 vx, vy）
	velocities0 = np.array(
		[
			[1.20, 0.35],
			[-0.55, 1.05],
			[1.35, -0.40],
		]
	)
	# 改这里：三体质量
	masses = np.array([0.5, 0.4, 0.6])
	return positions0, velocities0, masses


def plot_trajectories(trajectory):
	fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
	colors = ["tab:blue", "tab:orange", "tab:green"]
	labels = ["Body 1", "Body 2", "Body 3"]

	for i in range(3):
		ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], color=colors[i], label=labels[i])
		ax.scatter(trajectory[0, i, 0], trajectory[0, i, 1], color=colors[i], s=30)

	ax.set_aspect("equal", adjustable="box")
	ax.set_title("Three-Body Trajectories")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.legend()
	return fig, ax


def animate_trajectories(times, trajectory):
	fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
	colors = ["tab:blue", "tab:orange", "tab:green"]
	labels = ["Body 1", "Body 2", "Body 3"]

	ax.set_aspect("equal", adjustable="box")
	ax.set_title("Three-Body Motion")
	ax.set_xlabel("x")
	ax.set_ylabel("y")

	all_x = trajectory[:, :, 0]
	all_y = trajectory[:, :, 1]
	margin = 0.2
	xmin, xmax = all_x.min(), all_x.max()
	ymin, ymax = all_y.min(), all_y.max()
	span = max(xmax - xmin, ymax - ymin)
	center_x = 0.5 * (xmin + xmax)
	center_y = 0.5 * (ymin + ymax)
	ax.set_xlim(center_x - 0.5 * span - margin, center_x + 0.5 * span + margin)
	ax.set_ylim(center_y - 0.5 * span - margin, center_y + 0.5 * span + margin)

	trails = [ax.plot([], [], color=colors[i], lw=1.8)[0] for i in range(3)]
	points = [ax.plot([], [], marker="o", color=colors[i], ms=7, label=labels[i])[0] for i in range(3)]
	time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
	ax.legend(loc="upper right")

	def init():
		for line in trails + points:
			line.set_data([], [])
		time_text.set_text("")
		return trails + points + [time_text]

	def update(frame):
		for i in range(3):
			trails[i].set_data(trajectory[: frame + 1, i, 0], trajectory[: frame + 1, i, 1])
			# 动画显示当前质点位置，保持列表形式可避免部分后端不刷新
			points[i].set_data([trajectory[frame, i, 0]], [trajectory[frame, i, 1]])
		time_text.set_text(f"t = {times[frame]:.2f}")
		return trails + points + [time_text]

	ani = FuncAnimation(
		fig,
		update,
		frames=len(times),
		init_func=init,
		interval=20,
		blit=True,
	)
	return fig, ani


def main():
	# 改这里：切换或修改初始条件
	positions0, velocities0, masses = chaotic_initial_conditions()
	# 改这里：数值积分时间步长（越小越精细但越慢）
	dt = 0.002
	# 改这里：积分步数（越大模拟时间越长）
	steps = 60000
	times, trajectory = simulate(positions0, velocities0, masses, dt, steps)

	plot_trajectories(trajectory)
	# 这行要保留，避免动画对象被回收导致窗口不动
	fig, ani = animate_trajectories(times, trajectory)
	plt.show()


if __name__ == "__main__":
	main()
