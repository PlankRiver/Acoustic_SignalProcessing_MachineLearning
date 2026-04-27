import numpy as np
import matplotlib.pyplot as plt


# How to solve dN/dt = -lamda*N
N0 = 1000
N = N0
lamda = 0.1
dt = 1
t = 0.0
N_RK4 = []
time = []

N_RK4.append(N0)
time.append(t)
for i in range(100):
    k1 = -lamda*N
    k2 = -lamda*(N+k1*dt/2)
    k3 = -lamda*(N+k2*dt/2)
    k4 = -lamda*(N+k3*dt)
    k = (k1+2*k2+2*k3+k4)/6
    N += k*dt
    t += dt
    N_RK4.append(N)
    time.append(t)

time = np.array(time)
N_RK4 = np.array(N_RK4)
N_precise = N0*np.exp(-lamda*time)

plt.plot(time,N_RK4,'ro',label='RK4 Algorithm')
plt.plot(time,N_precise,'k-',label='The precise graph of Nuclear Decay')
plt.xlabel('Time')
plt.ylabel('Number of Nuclears')
plt.title('The solve of Nuclear_RK4')
plt.grid()
plt.show()
