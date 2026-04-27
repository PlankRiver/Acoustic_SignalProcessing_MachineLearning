import matplotlib.pyplot as plt
import numpy as np
import random

#折线图
# plt.plot([3,8,4,6,1,2,0,7])
# plt.show()
# plt.plot(range(7),[3,8,4,6,5,1,2])
# plt.show()

#plt.figure()
# plt.figure(
#     num=None,           # 图形编号或名称（数字或字符串）
#     figsize=None,       # 图形大小（宽, 高），单位英寸，如 (8, 6)
#     dpi=None,          # 分辨率（每英寸点数，默认100）
#     facecolor=None,    # 背景颜色（如 'white'、'lightgray'）
#     edgecolor=None,    # 边框颜色
#     frameon=True,      # 是否显示边框
#     clear=False        # 是否清空现有图形
# )
# t = np.arange(0.,4.,0.5)
# plt.plot(t,t,t**2,t+2,np.exp(t))
# plt.show()

#饼图与条状图
# plt.bar(range(7),[3,5,7,1,2,8,4])
# plt.show()
# plt.pie([3,5,7,1,2,8,4])
# plt.show()

#颜色与样式
# plt.plot(range(7),range(7)[::-1],'g--')
# plt.show()
# plt.plot(range(7),range(7)[::-1],'rD')
# plt.show()
# plt.figure(figsize=(8,6),dpi=100)
# t = np.arange(0.,4.,0.1)
# plt.plot(t,t,color='red',linestyle='-',linewidth=3,label='line1')
# plt.plot(t,t+2,linestyle='',marker='*',linewidth=3,label='line2')
# plt.plot(t,t**2,linestyle='',marker='+',linewidth=3,label='line3')
# plt.legend(loc='upper left')
# plt.show()
# plt.title('Plot Example')
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.plot(range(7),[3,6,4,5,9,1,2])
# plt.show()

#抛硬币
# times=10000
# result,resultmean=[],[]
# for i in range(times):
#     for k in range(10):
#         result.append(random.randint(0,1))
#     resultmean.append(np.mean(result))
# plt.plot(range(times),resultmean)
# plt.title('Coincidence')
# plt.xlabel('times')
# plt.ylabel('percentage')
# plt.show()

#subplot
# # 方法1：直接使用 plt.subplot()
# plt.subplot(nrows, ncols, index, **kwargs)
# # 方法2：面向对象方式（推荐）
# fig, ax = plt.subplots(nrows, ncols, **kwargs)
# plt.figure(1)
# plt.subplot(211)
# plt.plot(range(7),[3,7,4,5,9,1,8],color='red',marker='o')
# plt.subplot(212)
# plt.plot(range(7),range(7)[::-1],color='blue',marker='*')
# plt.show()

fig,(ax0,ax1) = plt.subplots(2,1)
ax0.plot(range(7),[5,8,6,7,3,4,1])
ax0.set_title('Plot A')
ax1.plot(range(7),[5,6,8,5,1,6,2])
ax1.set_title('Plot B')
plt.show()

# ax = plt.axes()
# rect：一个四元列表 [left, bottom, width, height]，定义坐标轴的位置和大小（取值范围 [0, 1]）。
