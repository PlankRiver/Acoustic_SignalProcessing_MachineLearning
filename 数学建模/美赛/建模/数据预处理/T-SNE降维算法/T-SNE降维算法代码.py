import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


def tsne_visualize(df, labels=None, perplexity=30, n_iter=1000):
    # 1. 标准化
    x = StandardScaler().fit_transform(df)

    # 2. 预降维 (加速 t-SNE)
    if x.shape[1] > 50:
        x = PCA(n_components=50).fit_transform(x)

    # 3. 执行 t-SNE
    # 注意：sklearn 1.5+ 版本中 n_iter 已改为 max_iter
    try:
        # 尝试新版本参数名
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    max_iter=n_iter,
                    init='pca',
                    learning_rate='auto',  # 建议显式设置，新版本默认是 auto
                    random_state=42)
    except TypeError:
        # 如果你的版本较旧，则回退到 n_iter
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    init='pca',
                    learning_rate='auto',
                    random_state=42)

    x_tsne = tsne.fit_transform(x)

    # 4. 绘图
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap='viridis', s=15)
        plt.colorbar(scatter)
    else:
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=15)

    plt.title(f"t-SNE Visualization (Perplexity={perplexity})")
    plt.show()

    return x_tsne


# 运行示例
digits = load_digits()
tsne_results = tsne_visualize(digits.data, labels=digits.target)