import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 1. 构造模拟数据
# n_samples: 样本数, centers: 实际簇数, n_features: 特征数
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. 数据预处理 (K-means 对量纲极其敏感，必须标准化)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 确定 K 值 (使用手肘法 Elbow Method)
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_) # inertia_ 即 SSE

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), sse, 'bo-')
plt.xlabel('K 值')
plt.ylabel('SSE (簇内误差平方和)')
plt.title('手肘法确定最优 K')

# 4. 执行 K-means 聚类
optimal_k = 4 # 假设从手肘图观察到 4 是拐点
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 5. 可视化结果
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=30, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X', label='质心')
plt.title(f"K-means 聚类结果 (K={optimal_k})")
plt.legend()
plt.tight_layout()
plt.show()