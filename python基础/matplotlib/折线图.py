from matplotlib import pyplot as plt
import random
# x = range(2,26,2)
# y = [4,7,1,3,6,9,2,13,12,9,7,8]
#
# #设置图片大小
# plt.figure(figsize=(10,8),dpi=200)
# plt.xticks(x)
# plt.yticks(range(min(y),max(y)+1))
# #绘图
# plt.plot(x,y)
#
# #show
# plt.show()





#气温曲线
temperature1 = [random.randint(20,35) for _ in range(120)]
temperature2 = [random.randint(20,35) for _ in range(120)]
time = range(120)
times = [f'10:{i:02d}' for i in range(60)]
times += [f'11:{i:02d}' for i in range(60)]

plt.figure(figsize=(20,8),dpi=200)
plt.xticks(time[::10], times[::10],rotation=45)
plt.grid(alpha=0.3,linestyle=':')  #透明度

plt.title('Temperature-Time Graph',color='green')
plt.xlabel('Time',color='red')
plt.ylabel('Temperature',color='blue')

plt.plot(time,temperature1,label='Temperature 1',color='red',linestyle='--',linewidth=2)
plt.plot(time,temperature2,label='Temperature 2',color='blue',linestyle='-',linewidth=3) #-实线 --虚线 -.点划线 ：点虚线
plt.legend(loc='upper left')

plt.show()