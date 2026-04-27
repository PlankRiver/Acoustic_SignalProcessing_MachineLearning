import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

# 1. 加载数据 (使用乳腺癌诊断数据集)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 划分训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立决策树模型 (使用 CART 算法)
# 关键参数：
# criterion: 'gini' 或 'entropy'
# max_depth: 限制深度，防止过拟合
# min_samples_leaf: 叶子节点最少样本数
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 4. 模型评估
y_pred = clf.predict(X_test)
print("-" * 30)
print("分类报告:\n", classification_report(y_test, y_pred))

# 5. 可视化决策树 (这是决策树论文的加分项)
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("决策树判定逻辑可视化")
plt.show()

# 6. 特征重要性分析 (揭示哪些指标最重要)
importances = clf.feature_importances_
feat_importances = pd.Series(importances, index=data.feature_names)
feat_importances.nlargest(5).plot(kind='barh')
plt.title("前 5 个最关键特征")
plt.show()