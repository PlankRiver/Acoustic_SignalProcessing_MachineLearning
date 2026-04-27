import numpy as np
import math
import matplotlib.pyplot as plt


# 1. 准备数据：计算距离矩阵
def calculate_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(locations[i] - locations[j])
    return dist_matrix


# 2. 计算路径总长度
def get_total_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    # 回到起点
    distance += dist_matrix[route[-1]][route[0]]
    return distance


# 3. 模拟退火核心算法
def simulated_annealing_tsp(locations, temp=1000, cooling_rate=0.995, stop_temp=1e-5):
    """
    :param locations: 城市坐标列表
    :param temp: 初始温度
    :param cooling_rate: 降温系数
    """
    n = len(locations)
    dist_matrix = calculate_distance_matrix(locations)

    # 初始化随机路径
    current_route = list(range(n))
    np.random.shuffle(current_route)

    current_dist = get_total_distance(current_route, dist_matrix)

    # 记录最优解
    best_route = list(current_route)
    best_dist = current_dist

    scores = []  # 用于画收敛图

    while temp > stop_temp:
        # --- 扰动产生新解 (交换两个城市的位置) ---
        new_route = list(current_route)
        i, j = np.random.randint(0, n, 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_dist = get_total_distance(new_route, dist_matrix)

        # --- Metropolis 准则 ---
        # 如果新路径更短，肯定接受
        # 如果新路径更长，以一定概率接受 (防止陷入局部最优)
        if new_dist < current_dist or np.random.rand() < math.exp((current_dist - new_dist) / temp):
            current_route = new_route
            current_dist = new_dist

            # 更新历史最优
            if current_dist < best_dist:
                best_dist = current_dist
                best_route = list(current_route)

        scores.append(best_dist)
        temp *= cooling_rate  # 降温

    return best_route, best_dist, scores


# ================= 使用示例 =================
if __name__ == "__main__":
    # 随机生成 20 个城市坐标
    np.random.seed(42)
    cities = np.random.rand(20, 2) * 100

    best_path, min_dist, score_history = simulated_annealing_tsp(cities)

    print("-" * 30)
    print(f"最短路径长度: {min_dist:.2f}")
    print(f"访问顺序: {best_path} -> {best_path[0]}")  # 闭环