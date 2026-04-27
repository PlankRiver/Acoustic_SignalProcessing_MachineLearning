import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca_analysis(df, n_components=0.95):
    """
    PCA分析
    :param n_components: 如果是整数，表示保留主成分个数；
                         如果是小数(0.95)，表示保留信息量的比例。
    """
    # 1. 标准化 (PCA必须做)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df)

    # 2. 建立PCA模型
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x_scaled)

    # 3. 特征值(方差)和贡献率
    explained_variance = pca.explained_variance_  # 特征值
    explained_variance_ratio = pca.explained_variance_ratio_  # 贡献率

    print(f"保留的主成分个数: {pca.n_components_}")
    print(f"各主成分贡献率: {explained_variance_ratio}")
    print(f"累计贡献率: {np.sum(explained_variance_ratio):.4f}")

    # 4. 主成分载荷矩阵 (解释主成分的含义)
    # 它是原始指标与主成分的相关系数
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(pca.n_components_)],
        index=df.columns
    )

    return principal_components, loadings


# --- 示例 ---
# 假设有5个指标的数据
data = np.random.rand(100, 5)
df = pd.DataFrame(data, columns=['指标A', '指标B', '指标C', '指标D', '指标E'])

pc_scores, pc_loadings = pca_analysis(df)
print("\n主成分载荷矩阵 (部分):")
print(pc_loadings.head())