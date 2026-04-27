# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

PI=3.1415927
h=0.02
v=1
tau=0.2
A=v*h/tau
NX = 50
NT = 4000
Y = np.zeros([NX+1,NT+1],dtype=float)
Time = np.arange(0,NT+1,1)
X = np.arange(0,(NX+1)*h,h)

for i in np.arange(NX+1): #初始条件
    Y[i,0]=np.sin(PI*i*h)   #y(x,t) = sin(pix)
#   Y[i,0]=np.sin(2*PI*i*h) #y(x,t) = sin(2pix)
    Y[i,1]= Y[i,0] #dy(x,t)/dt=0.0
for k in range(NT+1): #边界条件
    Y[0,k]=0.0
    Y[NX,k]=0.0

for k in range(1,NT):
    for i in range(1,NX):
        Y[i,k+1]=2.0*(1-A**2)*Y[i,k]+A**2*(Y[i+1,k]+Y[i-1,k])-Y[i,k-1]

fig = plt.figure(figsize=(10,4))
ax1 =fig.add_subplot(1,2,1)
ax2 =fig.add_subplot(1,2,2)

ax1.plot(X,Y[:,0], 'r-', label='t=0',linewidth=1.0)
ax1.plot(X,Y[:,200], 'b-', label='t=200',linewidth=1.0)
ax1.plot(X,Y[:,300], 'g-', label='t=300',linewidth=1.0)
ax1.plot(X,Y[:,400], 'k-', label='t=400',linewidth=1.0)
ax1.set_ylabel(r'Y', fontsize=20)
ax1.set_xlabel(r'X', fontsize=20)
ax1.set_xlim(0,1)
ax1.set_ylim(-2,2)
ax1.legend(loc='upper right',fontsize=10)

extent = [0,250,0,50]
levels = np.arange(-2.0,2.0,0.01)
CS = ax2.contourf(Y,levels,origin='lower',extent=extent,cmap=plt.cm.hot)
ax2.set_ylabel(r'X', fontsize=15)
ax2.set_xlabel(r'Time', fontsize=15)
plt.show()
