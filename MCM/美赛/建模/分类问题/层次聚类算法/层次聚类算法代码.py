import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 1. 构造模拟数据
X, y = make_blobs(n_samples=50, centers=3, random_state=42)

# 2. 标准化 (重要：距离敏感算法必须标准化)
X_scaled = StandardScaler().fit_transform(X)

# 3. 计算层次聚类矩阵 (使用 Ward 方法)
# 'ward' 为联动准则，'euclidean' 为距离度量
Z = linkage(X_scaled, method='ward', metric='euclidean')

# 4. 绘制树状图 (Dendrogram) —— 层次聚类的灵魂
plt.figure(figsize=(12, 6))
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.title('层次聚类树状图 (Dendrogram)')
plt.xlabel('样本索引')
plt.ylabel('Ward 距离')

# 画一条水平阈值线，辅助确定分类数 (比如在距离 7 处切开)
plt.axhline(y=7, color='r', linestyle='--')
plt.show()

# 5. 根据预设类数或距离阈值提取聚类结果
# 方案 A: 设定固定类数为 3
max_d = 7
clusters = fcluster(Z, t=3, criterion='maxclust')
print(f"前10个样本的聚类结果: {clusters[:10]}")