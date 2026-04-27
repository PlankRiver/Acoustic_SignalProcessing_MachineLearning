# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

N=50
Nhalf = N//2
Nt =3000
tau = 1.0 #时间步长
h = 1.0 #空间步长
Du, Dv, f, r = 0.16, 0.08, 0.060, 0.062
#Du, Dv, f, r = 0.14, 0.06, 0.035, 0.065

Au = Du*tau/h**2
Av = Dv*tau/h**2

U = np.zeros((N+1, N+1, Nt+1), dtype=float)
V = np.zeros((N+1, N+1, Nt+1), dtype=float)

# initialization
U[:,:,0] = U[:,:,0] + 0.02*np.random.random((N+1,N+1))
V[:,:,0] = V[:,:,0] + 0.02*np.random.random((N+1,N+1))
U[Nhalf-16:Nhalf+16,Nhalf-16:Nhalf+16,0] = 0.50
V[Nhalf-16:Nhalf+16,Nhalf-16:Nhalf+16,0] = 0.25

for k in range(Nt):
    for i in range(1,N):
        for j in range(1,N):

            U[i,j,k+1] = (1-4.0*Au)*U[i,j,k]+ \
                Au*(U[i+1,j,k]+U[i-1,j,k]+U[i,j+1,k]+U[i,j-1,k])-\
                tau*U[i,j,k]*V[i,j,k]*V[i,j,k]+tau*f*(1.0-U[i,j,k])

            V[i,j,k+1] = (1-4.0*Av)*V[i,j,k]+ \
                Av*(V[i+1,j,k]+V[i-1,j,k]+V[i,j+1,k]+V[i,j-1,k])+\
                tau*U[i,j,k]*V[i,j,k]*V[i,j,k]-tau*(f+r)*V[i,j,k]
    for i in range(1,N):
        U[i,0,k+1] = U[i,N-1,k+1]
        U[i,N,k+1] = U[i,1,k+1]
        V[i,0,k+1] = V[i,N-1,k+1]
        V[i,N,k+1] = V[i,1,k+1]
    for j in range(0,N+1):
        U[0,j,k+1] = U[N-1,j,k+1]
        U[N,j,k+1] = U[1,j,k+1]
        V[0,j,k+1] = V[N-1,j,k+1]
        V[N,j,k+1] = V[1,j,k+1]

fig = plt.figure(figsize=(10,4))
ax1 =fig.add_subplot(1,2,1)
ax2 =fig.add_subplot(1,2,2)
#plt.axis('tight')
levels = np.arange(0.0,1.0,0.01)

#ax1.pcolor(U[:,:,Nt].reshape((N+1, N+1)), vmin=0.0, vmax=1.0,cmap=plt.cm.RdBu)
#ax2.pcolor(V[:,:,Nt].reshape((N+1, N+1)), vmin=0.0, vmax=1.0, cmap=plt.cm.RdBu)

ax1.pcolor(U[:,:,Nt].reshape((N+1, N+1)),cmap=plt.cm.RdBu)
ax2.pcolor(V[:,:,Nt].reshape((N+1, N+1)),cmap=plt.cm.RdBu)
plt.show()
