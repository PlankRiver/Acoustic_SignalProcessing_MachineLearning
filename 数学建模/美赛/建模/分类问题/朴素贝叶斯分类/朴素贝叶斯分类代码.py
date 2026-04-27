import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine

# 1. 加载数据 (使用葡萄酒质量分类数据集)
data = load_wine()
X, y = data.data, data.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 建立高斯朴素贝叶斯模型
# 朴素贝叶斯几乎没有超参数需要调，非常稳定
model = GaussianNB()
model.fit(X_train, y_train)

# 4. 预测与评估
y_pred = model.predict(X_test)
# 获取预测的概率 (这是贝叶斯的强项)
y_prob = model.predict_proba(X_test)

print("-" * 30)
print("分类报告:\n", classification_report(y_test, y_pred))
print(f"测试集前 5 个样本的预测概率:\n{y_prob[:5]}")

# 5. 可视化混淆矩阵
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("混淆矩阵")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.show()