import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


def umap_visualize(data, labels=None, n_neighbors=15, min_dist=0.1):
    """
    UMAP 降维可视化
    :param n_neighbors: 邻居数（平衡局部与全局结构，越大越全局）
    :param min_dist: 点与点之间的最小距离（越大点越分散）
    """
    # 1. 标准化
    scaled_data = StandardScaler().fit_transform(data)

    # 2. 运行 UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=42)
    embedding = reducer.fit_transform(scaled_data)

    # 3. 绘图
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)

    plt.title(f"UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})")
    plt.show()

    return embedding


# --- 示例：使用手写数字数据集 ---
digits = load_digits()
umap_results = umap_visualize(digits.data, labels=digits.target)