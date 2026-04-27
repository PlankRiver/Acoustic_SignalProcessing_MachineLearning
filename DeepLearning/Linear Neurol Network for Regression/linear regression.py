import torch
import matplotlib.pyplot as plt
import numpy as np
import math


#normalization
def normal(x, mu, sigma):
    p = 1/math.sqrt(2*math.pi*sigma**2)
    return p*np.exp(-(x-mu)**2/(2*sigma**2))

x = np.linspace(-10, 10, 10000)
y1 = normal(x, 0, 1)
y2 = normal(x, 0, 2)
y3 = normal(x, 3, 1)

plt.title('Normalization Function',color='green')
plt.xlabel('X-axis',color='red')
plt.ylabel('Y-axis',color='blue')
plt.plot(x,y1,label='mu=0,std=1',color='red',linestyle='--',linewidth=2)
plt.plot(x,y2,label='mu=0,std=2',color='blue',linestyle='-',linewidth=3)
plt.plot(x,y3,label='mu=3,std=1',color='green',linestyle='--',linewidth=3)
plt.legend(loc='upper left')
plt.show()