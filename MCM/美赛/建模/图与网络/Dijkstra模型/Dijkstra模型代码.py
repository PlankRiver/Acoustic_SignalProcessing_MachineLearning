import heapq


def dijkstra(graph, start):
    """
    Dijkstra 算法实现
    :param graph: 字典形式的邻接表, e.g., {'A': {'B': 5, 'C': 1}}
    :param start: 起点
    :return: distances(最短距离字典), predecessors(前驱节点字典)
    """
    # 1. 初始化
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}

    # 优先队列，存储 (距离, 节点)
    priority_queue = [(0, start)]

    while priority_queue:
        # 2. 选取当前距离最小的节点
        current_dist, u = heapq.heappop(priority_queue)

        # 如果弹出的距离已经大于记录的距离，说明是旧数据，跳过
        if current_dist > distances[u]:
            continue

        # 3. 遍历邻居进行松弛操作
        for v, weight in graph[u].items():
            distance = current_dist + weight

            # 4. 松弛公式: 如果通过 u 到达 v 的距离更短
            if distance < distances[v]:
                distances[v] = distance
                predecessors[v] = u
                heapq.heappush(priority_queue, (distance, v))

    return distances, predecessors


def get_path(predecessors, start, end):
    """回溯路径"""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path if path[0] == start else None


# ================= 使用示例 =================

if __name__ == "__main__":
    # 定义网络结构 (邻接表)
    network = {
        'A': {'B': 4, 'C': 2},
        'B': {'C': 3, 'D': 2, 'E': 3},
        'C': {'B': 1, 'D': 4, 'E': 5},
        'D': {'E': 1},
        'E': {}
    }

    start_node = 'A'
    end_node = 'E'

    dists, parents = dijkstra(network, start_node)
    path = get_path(parents, start_node, end_node)

    print("-" * 30)
    print(f"从 {start_node} 到 {end_node} 的最短距离: {dists[end_node]}")
    print(f"具体路径: {' -> '.join(path)}")