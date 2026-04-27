# 如果没有安装，请先 pip install joypy
import joypy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. 模拟数据生成
# 假设有10个样本(Samp A-J)，每个样本有1000个观测值
data = pd.DataFrame()
samples = ['Samp ' + chr(i) for i in range(ord('A'), ord('K'))]
for s in samples:
    # 生成带随机偏移的分布数据
    mu = np.random.uniform(30, 60)
    vals = np.random.normal(mu, 10, 1000)
    temp_df = pd.DataFrame({'Value': vals, 'Sample': s})
    data = pd.concat([data, temp_df])

# 2. 绘制山脊图
fig, axes = joypy.joyplot(data, by="Sample", column="Value",
                          colormap=cm.Spectral_r,  # 使用类似原图的彩色映射
                          fade=True,              # 渐淡效果
                          range_style='own',      # 每个分布使用自己的X轴范围或统一
                          grid=True,
                          linewidth=1,
                          legend=False,
                          figsize=(10, 7),
                          title="Ridgeline Plot")

plt.xlabel("K (w)")
plt.show()