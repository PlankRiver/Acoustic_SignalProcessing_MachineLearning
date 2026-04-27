import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. 加载数据 (使用经典的鸢尾花数据集)
iris = load_iris()
X, y = iris.data, iris.target

# 2. 标准化 (KNN 极度依赖距离，必须标准化，否则数值大的特征会主导结果)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 寻找最优 K 值 (使用交叉验证)
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 使用 10 折交叉验证
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# 可视化 K 值选择
plt.plot(k_range, k_scores, 'go-')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('手肘法/交叉验证寻找最优 K')
plt.show()

# 5. 建立最终模型
optimal_k = k_range[np.argmax(k_scores)]
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train, y_train)

# 6. 评估
print("-" * 30)
print(f"确定的最优 K 值: {optimal_k}")
print(f"测试集准确率: {knn_final.score(X_test, y_test):.4f}")