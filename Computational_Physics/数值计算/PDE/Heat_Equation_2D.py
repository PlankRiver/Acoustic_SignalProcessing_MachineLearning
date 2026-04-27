# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

tau = 0.05 #时间步长
h = 0.5 #空间步长
lamda = 1.0
A = 1.0*tau/ h**2
l = 10.0 #开长
NX = int(l/h) #空间节点数
NY = int(l/h) #空间节点数
NT = 1000 #时间步数

U = np.zeros([NX+1,NX+1,NT+1])  #温度
t = np.arange(0,NT,1)  #时间
x = np.arange(0,(NX+1)*h,h) #坐标
y = np.arange(0,(NY+1)*h,h) #坐标

for i in range(0,NX): #初始条件
    for j in range(0,NY):
        U[i,j,0] = 0.0

for k in range(0,NT):
    for i in range(1,NX):
        for j in range(1,NY):
            U[i,j,k+1] = (1-4.0*A)*U[i,j,k]+\
                A*(U[i+1,j,k]+U[i-1,j,k]+U[i,j+1,k]+U[i,j-1,k])+\
                tau*5.0*np.exp(-2.0*((i*h-5.0)**2+(j*h-5.0)**2))

    for i in range(0,NX+1):  #第二类边界条件
        U[i,0,k+1] = U[i,1,k+1]
        U[i,NX,k+1] = U[i,NX-1,k+1]
    for j in range(0,NY+1):  #第一类边界条件
        U[0,j,k+1] = 0.0
        U[NX,j,k+1] = 0.0

fig = plt.figure(figsize=(10,10))
ax1 =fig.add_subplot(2,2,1)
ax2 =fig.add_subplot(2,2,2)
ax3 =fig.add_subplot(2,2,3)
ax4 =fig.add_subplot(2,2,4)

extent = [0,(NX+1)*h,0,(NY+1)*h]
levels = np.arange(0.0,3.5,0.01)
cs1 = ax1.contourf(U[:,:,10].T,levels,origin='lower',
                extent=extent,cmap=plt.cm.hot,label='NT=10')
cs2 = ax2.contourf(U[:,:,100].T,levels,origin='lower',extent=extent,cmap=plt.cm.hot)
cs3 = ax3.contourf(U[:,:,300].T,levels,origin='lower',extent=extent,cmap=plt.cm.hot)
cs4 = ax4.contourf(U[:,:,1000].T,levels,origin='lower',extent=extent,cmap=plt.cm.hot)

ax1.set_ylabel('rY', fontsize=15)
ax3.set_xlabel('rX', fontsize=15)
ax3.set_ylabel('rY', fontsize=15)
ax4.set_xlabel('rX', fontsize=15)
ax1.set_title('NT=10')
ax2.set_title('NT=100')
ax3.set_title('NT=300')
ax4.set_title('NT=1000')
plt.show()
