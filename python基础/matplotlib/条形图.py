import matplotlib.pyplot as plt
import random

a = ['Lord of\n the ring','Lion\n King','Killer','Game of\n throne','Dog']
b = [91.4,87.6,78.2,90.0,84.5]

#单个条形图
# plt.figure(figsize=(10,10))
# plt.xticks(range(len(a)),a,rotation=45)
# plt.title('Rate of movies')
# plt.xlabel('Movies',color='blue')
# plt.ylabel('Rate',color='red')
# plt.bar(range(len(a)),b,color='red',width=0.3)     #横图plt.barh(a,b,height=0.3) 坐标交换
# plt.show()


#多个条形图
b_14 = b
b_15 = [43.9,23.8,54.5,37.7,72.2]
b_16 = [(x+y)/2 for x,y in zip(b_14,b_15)]

x_14 = range(len(a))
x_15 = [i+0.2 for i in x_14]
x_16 = [i+0.4 for i in x_14]

plt.xticks(x_15,a,rotation=45)
plt.bar(x_14,b_14,color='red',label='September 14th',width=0.2)
plt.bar(x_15,b_15,color='blue',label='September 15th',width=0.2)
plt.bar(x_16,b_16,color='green',label='September 16th',width=0.2)
plt.legend(loc='upper right')

plt.show()