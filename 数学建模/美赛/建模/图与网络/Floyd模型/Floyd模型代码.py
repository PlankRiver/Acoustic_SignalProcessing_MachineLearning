import numpy as np


def floyd_warshall(matrix):
    """
    Floyd 算法实现
    :param matrix: 初始邻接矩阵 (二维数组)，不连通设为 np.inf
    :return: 最终距离矩阵, 路径矩阵
    """
    n = len(matrix)
    # 拷贝一份矩阵，避免修改原始数据
    dist = np.array(matrix, dtype=float)

    # 初始化路径矩阵 (记录从 i 到 j 必须经过的最后一个节点)
    path = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] != np.inf:
                path[i][j] = j
            else:
                path[i][j] = -1

    # 三重循环：k 必须在外层
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # 状态转移：判断经过 k 是否更短
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = path[i][k]  # 更新路径

    return dist, path


def print_path(path, i, j):
    """递归打印路径"""
    if path[i][j] == -1:
        return "没有路径"
    route = [i]
    while i != j:
        i = path[i][j]
        route.append(i)
    return " -> ".join(map(str, route))


# ================= 使用示例 =================

if __name__ == "__main__":
    INF = np.inf
    # 定义邻接矩阵
    adj_matrix = [
        [0, 3, 8, INF, -4],
        [INF, 0, INF, 1, 7],
        [INF, 4, 0, INF, INF],
        [2, INF, -5, 0, INF],
        [INF, INF, INF, 6, 0]
    ]

    dist_matrix, path_matrix = floyd_warshall(adj_matrix)

    print("-" * 30)
    print("全源最短距离矩阵:")
    print(dist_matrix)

    print("\n从节点 0 到 2 的最短路径:")
    print(print_path(path_matrix, 0, 2))