import pandas as pd

data = [10, 15, 12, 18, 25, 30, 80] # 80是异常值
# 划分为 3 个等宽区间
bins_width = pd.cut(data, bins=3)
print(f"等宽离散化结果:\n{bins_width.value_counts()}")

# 划分为 3 个等频区间 (利用 qcut)
bins_freq = pd.qcut(data, q=3)
print(f"等频离散化结果:\n{bins_freq.value_counts()}")

from sklearn.cluster import KMeans

def cluster_binning(data, k=3):
    data_res = np.array(data).reshape(-1, 1)
    k_means = KMeans(n_clusters=k, random_state=42).fit(data_res)
    centers = sorted(k_means.cluster_centers_.flatten())
    # 计算簇间中点作为边界
    thresholds = [(centers[i] + centers[i+1])/2 for i in range(len(centers)-1)]
    return thresholds

data = [1, 2, 3, 10, 11, 12, 25, 26, 27]
print(f"聚类确定的离散化边界: {cluster_binning(data, k=3)}")
# 输出约等于 [6.5, 18.5]，完美切分了三群数据