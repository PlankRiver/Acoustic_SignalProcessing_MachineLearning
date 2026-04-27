import numpy as np
import matplotlib.pyplot as plt


def equilateral_triangle(L):
    return np.array(
        [
            [0.0, np.sqrt(3.0) * L / 3.0],
            [-L / 2.0, -np.sqrt(3.0) * L / 6.0],
            [L / 2.0, -np.sqrt(3.0) * L / 6.0],
        ],
        dtype=float,
    )


def rhs(pos, v):
    targets = pos[[1, 2, 0]]
    direction = targets - pos
    distance = np.linalg.norm(direction, axis=1, keepdims=True)
    distance = np.maximum(distance, 1e-12)
    return v * direction / distance


def rk4_step(pos, dt, v):
    k1 = rhs(pos, v)
    k2 = rhs(pos + 0.5 * dt * k1, v)
    k3 = rhs(pos + 0.5 * dt * k2, v)
    k4 = rhs(pos + dt * k3, v)
    return pos + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate(L=1.0, v=1.0, dt=1e-3, max_steps=20000, tol=1e-4):
    pos = equilateral_triangle(L)
    center = pos.mean(axis=0)

    trajectory = [pos.copy()]
    distances = [np.linalg.norm(pos - center, axis=1)]
    times = [0.0]

    t = 0.0
    for _ in range(max_steps):
        if np.max(np.linalg.norm(pos - center, axis=1)) < tol:
            break
        pos = rk4_step(pos, dt, v)
        t += dt
        trajectory.append(pos.copy())
        distances.append(np.linalg.norm(pos - center, axis=1))
        times.append(t)

    return np.array(times), np.array(trajectory), np.array(distances), center


def main():
    L = 1.0
    v = 1.0
    dt = 1e-3

    times, trajectory, distances, center = simulate(L=L, v=v, dt=dt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax1, ax2 = axes
    labels = ["Particle 1", "Particle 2", "Particle 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i in range(3):
        ax1.plot(trajectory[:, i, 0], trajectory[:, i, 1], color=colors[i], label=labels[i])

    ax1.scatter(
        [trajectory[0, 0, 0], trajectory[0, 1, 0], trajectory[0, 2, 0]],
        [trajectory[0, 0, 1], trajectory[0, 1, 1], trajectory[0, 2, 1]],
        color=colors,
        s=30,
    )
    ax1.scatter(center[0], center[1], color="black", s=25)
    ax1.set_title("Particle Trajectories")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend()

    ax2.plot(times, distances[:, 0], color="tab:blue")
    ax2.set_title("Particle 1 Distance to Center")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Distance to Center")

    plt.show()


if __name__ == "__main__":
    main()
