import numpy as np
import matplotlib.pyplot as plt


# 1. 定义目标函数 (Rastrigin Function)
def objective_function(x):
    # f(x) = sum(x_i^2 - 10*cos(2*pi*x_i) + 10)
    # 理论全局最小值在 x = [0,0...,0] 处，值为 0
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


class PSO:
    def __init__(self, n_particles, dimensions, max_iter, bounds):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.lb, self.ub = bounds  # 搜索范围上下界

        # 初始参数
        self.w = 0.8  # 惯性权重
        self.c1 = 2.0  # 个体学习因子
        self.c2 = 2.0  # 社会学习因子

        # 初始化粒子
        self.X = np.random.uniform(self.lb, self.ub, (n_particles, dimensions))
        self.V = np.random.uniform(-1, 1, (n_particles, dimensions))

        # 初始化极值
        self.pbest_X = copy.deepcopy(self.X)
        self.pbest_fit = np.array([objective_function(p) for p in self.X])

        self.gbest_X = self.pbest_X[np.argmin(self.pbest_fit)].copy()
        self.gbest_fit = np.min(self.pbest_fit)

        self.history = []

    def optimize(self):
        for t in range(self.max_iter):
            # 线性递减权重策略 (提高收敛性能)
            w_t = self.w - (self.w - 0.4) * (t / self.max_iter)

            for i in range(self.n_particles):
                # 1. 更新速度
                r1, r2 = np.random.rand(), np.random.rand()
                self.V[i] = (w_t * self.V[i] +
                             self.c1 * r1 * (self.pbest_X[i] - self.X[i]) +
                             self.c2 * r2 * (self.gbest_X - self.X[i]))

                # 2. 更新位置
                self.X[i] = self.X[i] + self.V[i]

                # 3. 边界处理
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)

                # 4. 更新个体最优
                fit = objective_function(self.X[i])
                if fit < self.pbest_fit[i]:
                    self.pbest_fit[i] = fit
                    self.pbest_X[i] = self.X[i].copy()

                    # 5. 更新全局最优
                    if fit < self.gbest_fit:
                        self.gbest_fit = fit
                        self.gbest_X = self.X[i].copy()

            self.history.append(self.gbest_fit)
            if t % 10 == 0:
                print(f"Iteration {t}: Best Fitness = {self.gbest_fit:.6f}")

        return self.gbest_X, self.gbest_fit


# ================= 运行示例 =================
import copy

if __name__ == "__main__":
    # 5个决策变量，每个范围在 [-5.12, 5.12]
    pso = PSO(n_particles=30, dimensions=5, max_iter=100, bounds=(-5.12, 5.12))
    best_x, best_fit = pso.optimize()

    print("-" * 30)
    print(f"全局最优解: {best_x}")
    print(f"最小函数值: {best_fit}")

    # 绘制收敛曲线
    plt.plot(pso.history)
    plt.title("PSO 收敛曲线")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值 (Fitness)")
    plt.grid(True)
    plt.show()