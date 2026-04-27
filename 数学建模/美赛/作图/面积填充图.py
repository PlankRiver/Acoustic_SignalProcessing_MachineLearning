# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 1. 设置学术风格
# sns.set_style("white") # 干净背景
# plt.rcParams['font.family'] = 'serif' # 衬线字体，美赛推荐
#
# # 2. 准备模拟数据 (实际套用时替换为你的结果数组)
# np.random.seed(42)
# data1 = np.random.normal(loc=1.5, scale=0.3, size=1000) # 橙色组
# data2 = np.random.normal(loc=1.8, scale=0.35, size=1000) # 蓝色组
# data3 = np.random.normal(loc=2.2, scale=0.4, size=1000) # 黄色组
#
# # 3. 创建画布
# fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
#
# # 4. 绘制密度图
# # fill=True: 填充颜色, alpha: 设置透明度, linewidth: 轮廓线宽
# sns.kdeplot(data1, fill=True, color="#F08080", alpha=0.6, label="Model A", linewidth=1.5)
# sns.kdeplot(data2, fill=True, color="#5B9BD5", alpha=0.5, label="Model B", linewidth=1.5)
# sns.kdeplot(data3, fill=True, color="#FFD966", alpha=0.4, label="Model C", linewidth=1.5)
#
# # 5. 美化细节
# ax.set_title('Probability Density Comparison', fontsize=15, fontweight='bold', pad=15)
# ax.set_xlabel('Value ($K_S$)', fontsize=12)
# ax.set_ylabel('Density', fontsize=12)
#
# # 移除上方和右侧边框 (学术论文标准样式)
# sns.despine()
#
# # 添加图例
# ax.legend(frameon=False, loc='upper right')
#
# plt.tight_layout()
# plt.show()




# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # 模拟数据
# data = np.random.normal(loc=110, scale=25, size=500)
#
# fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
#
# # 绘制直方图 (hist) 并叠加 KDE 曲线
# sns.histplot(data, kde=True, bins=30, color="#5B9BD5",
#              edgecolor='white', alpha=0.7, line_kws={'linewidth': 2, 'color': 'black'})
#
# # 模仿原图添加垂直参考线 (均值线)
# mean_val = np.mean(data)
# ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5)
# ax.axvline(mean_val - 20, color='gray', linestyle=':', linewidth=1)
# ax.axvline(mean_val + 20, color='gray', linestyle=':', linewidth=1)
#
# # 美化
# ax.set_xlabel('Time (Ma)', fontsize=12)
# ax.set_ylabel('Frequency', fontsize=12)
# sns.despine()
#
# plt.tight_layout()
# plt.show()