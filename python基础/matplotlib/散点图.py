import matplotlib.pyplot as plt
import random

y_3 = [random.randint(10,35) for _ in range(31)]
y_10 = [random.randint(10,35) for _ in range(31)]
x_3 = range(1,32)
x_10 = range(51,82)
_x = list(x_3)+list(x_10)
x_ticks = [f'3.{i:02d}' for i in x_3]
x_ticks += [f'10.{i-30:02d}' for i in x_10]
plt.figure(figsize=(20,8), dpi=80)

plt.xticks(_x[::3],x_ticks[::3],rotation=45)
plt.title('The temperature graph')
plt.xlabel('Time',color='blue')
plt.ylabel('Temperature',color='red')

plt.scatter(x_3,y_3,label='March',color='red')
plt.scatter(x_10,y_10,label='October',color='blue')
plt.legend(loc='upper left')

plt.show()