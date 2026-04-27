import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 构造数据
data = {
    '教育程度': ['高中']*50 + ['本科']*50 + ['硕士']*50,
    '年收入(w)': np.concatenate([
        np.random.normal(8, 2, 50),   # 高中：均值8
        np.random.normal(15, 4, 50),  # 本科：均值15
        np.random.normal(25, 6, 50)   # 硕士：均值25
    ])
}
df = pd.DataFrame(data)

# 2. 绘制箱线图
plt.figure(figsize=(10, 6))
# 使用 seaborn 绘制，美观且功能强大
sns.boxplot(x='教育程度', y='年收入(w)', data=df, palette="Set3")

# 3. 添加点图 (Swarmplot) 叠加在箱线图上，可以看到真实的数据分布密度
sns.swarmplot(x='教育程度', y='年收入(w)', data=df, color="0.25", size=3)

plt.title("教育程度与年收入的箱线图分析")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()