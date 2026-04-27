import numpy as np
import pandas as pd


def detect_outliers_3sigma(data):
    """
    3sigma原则检测异常值
    :param data: 一维数组或Series
    :return: 异常值的索引, 异常值本身
    """
    mean_val = np.mean(data)
    std_val = np.std(data)

    # 计算上下界
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val

    # 筛选异常值
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    return np.where(outliers_mask)[0], data[outliers_mask]


# --- 示例 ---
data = pd.Series([10, 12, 11, 13, 12, 11, 100, 11, 12, 11])  # 100 是明显的异常
idx, val = detect_outliers_3sigma(data)
print(f"3sigma检测到的异常值索引: {idx}, 数值: {val.values}")