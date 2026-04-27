import numpy as np
import matplotlib.pyplot as plt


# 1. 目标函数 (适应度函数)
def fitness_func(x):
    return x * np.sin(10 * np.pi * x) + 2.0


class GeneticAlgorithm:
    def __init__(self, bounds, pop_size=100, genome_len=20, n_generations=200, p_c=0.8, p_m=0.01):
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.n_generations = n_generations
        self.p_c = p_c  # 交叉概率
        self.p_m = p_m  # 变异概率

        # 初始化种群 (随机 0, 1 矩阵)
        self.pop = np.random.randint(0, 2, (pop_size, genome_len))

    def decode(self, pop):
        """将二进制编码转换为实数"""
        precision = (self.ub - self.lb) / (2 ** self.genome_len - 1)
        # 将矩阵的每一行从二进制转为十进制
        res = pop.dot(2 ** np.arange(self.genome_len)[::-1])
        return self.lb + res * precision

    def select(self, fitness):
        """轮盘赌选择"""
        # 确保适应度为正
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        """单点交叉"""
        if np.random.rand() < self.p_c:
            i_ = np.random.randint(0, self.pop_size, size=1)  # 选另一个
            cross_points = np.random.randint(0, 2, size=self.genome_len).astype(bool)
            parent[cross_points] = pop[i_, cross_points]
        return parent

    def mutate(self, child):
        """变异"""
        for i in range(self.genome_len):
            if np.random.rand() < self.p_m:
                child[i] = 1 if child[i] == 0 else 0
        return child

    def evolve(self):
        best_fitness_history = []

        for gen in range(self.n_generations):
            # 1. 解码并计算适应度
            x_values = self.decode(self.pop)
            fitness = fitness_func(x_values)

            # 记录本代最佳
            best_idx = np.argmax(fitness)
            best_fitness_history.append(fitness[best_idx])

            # 2. 选择
            self.pop = self.select(fitness)
            pop_copy = self.pop.copy()

            # 3. 交叉与变异
            for i in range(self.pop_size):
                child = self.crossover(self.pop[i], pop_copy)
                child = self.mutate(child)
                self.pop[i] = child

        return self.decode(self.pop), fitness, best_fitness_history


# ================= 运行示例 =================

if __name__ == "__main__":
    ga = GeneticAlgorithm(bounds=(-1, 2), n_generations=100)
    final_pop_x, final_fitness, history = ga.evolve()

    best_idx = np.argmax(final_fitness)
    print("-" * 30)
    print(f"全局最优解 x: {final_pop_x[best_idx]:.6f}")
    print(f"最大值 f(x): {final_fitness[best_idx]:.6f}")

    plt.plot(history)
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()