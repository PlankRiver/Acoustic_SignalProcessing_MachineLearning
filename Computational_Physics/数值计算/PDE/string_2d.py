# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

PI=3.1415927
h=PI/70
v=1
tau=0.2
A=v*h/tau
NX = 70
NY = 70
NT = 250
U = np.zeros([NX+1,NY+1,NT+1])
X = np.arange(0,NX+1,1)
Y = np.arange(0,NY+1,1)
for i in range(NX+1): #初始条件
    for j in range(NY+1):
        U[i,j,0]=3*np.sin(i*h)*np.sin(j*h) #u(x,t) = sin(x)sin(y)
        U[i,j,1]=3*np.sin(i*h)*np.sin(j*h) #du(x,y,t)/dt=0.0
for k in range(0,NT+1): #边界条件 y
    for i in range(NX+1):
        U[i,0,k]=0
        U[i,NY,k]=0
for k in range(0,NT+1): #边界条件 x
    for j in range(NY+1):
        U[0,j,k]=0
        U[NX,j,k]=0
for k in range(1,NT): #有限差分
    for i in range(1,NX):
        for j in range(1,NY):
            U[i,j,k+1]=2*U[i,j,k]-U[i,j,k-1]+A**2*(U[i+1,j,k]+U[i-1,j,k]-\
                4*U[i,j,k]+U[i,j+1,k]+U[i,j-1,k])
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(X, Y)
#Z = U[X,Y,5]
ax1.plot_wireframe(X,Y,U[:,:,200],color='r')
ax1.set_ylabel(r'Y', fontsize=20)
ax1.set_xlabel(r'X', fontsize=20)
ax1.set_zlabel(r'U(X,Y)', fontsize=20)
plt.show()
