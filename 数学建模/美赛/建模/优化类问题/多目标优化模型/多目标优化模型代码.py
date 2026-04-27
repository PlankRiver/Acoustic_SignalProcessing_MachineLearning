import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt


# --- 1. 定义多目标规划问题 ---
class MyMultiObjectiveProblem(ElementwiseProblem):
    def __init__(self):
        # n_var=2 (两个决策变量x1, x2)
        # n_obj=2 (两个目标函数f1, f2)
        # n_ieq_constr=2 (两个不等式约束)
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2,
                         xl=np.array([-2, -2]), xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        # 目标函数 1: f1 = x1^2 + x2^2
        f1 = x[0] ** 2 + x[1] ** 2
        # 目标函数 2: f2 = (x1-1)^2 + x2^2
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        # 约束条件 (要求 <= 0)
        # g1: x1 + x2 - 1 <= 0
        g1 = x[0] + x[1] - 1
        # g2: -x1 - x2 - 1 <= 0
        g2 = -x[0] - x[1] - 1

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# --- 2. 实例化算法 ---
problem = MyMultiObjectiveProblem()
algorithm = NSGA2(
    pop_size=40,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# --- 3. 运行优化 ---
termination = get_termination("n_gen", 100)  # 运行 100 代
res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=False)

# --- 4. 结果可视化 (绘制帕累托前沿) ---
F = res.F
plt.figure(figsize=(8, 6))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Pareto Front (帕累托前沿)")
plt.xlabel("Objective 1 (f1)")
plt.ylabel("Objective 2 (f2)")
plt.grid(True)
plt.show()

# --- 5. 选出一个最优解 (如利用满意度函数或简单取中值) ---
print(f"找到的帕累托解个数: {len(F)}")
print(f"其中一个代表性解 (目标函数值): {F[0]}")