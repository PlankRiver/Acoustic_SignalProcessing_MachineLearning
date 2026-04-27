import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def detect_outliers_boxplot(data):
    """
    箱线图方法检测异常值
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers_mask = (data < lower_limit) | (data > upper_limit)
    return np.where(outliers_mask)[0], data[outliers_mask]


# --- 可视化展示 ---
data = np.random.normal(10, 2, 100)
data = np.append(data, [25, -5])  # 手动添加异常点

plt.figure(figsize=(6, 4))
sns.boxplot(data=data)
plt.title("Boxplot Outlier Detection")
plt.show()

idx, val = detect_outliers_boxplot(data)
print(f"箱线图检测到的异常值: {val}")