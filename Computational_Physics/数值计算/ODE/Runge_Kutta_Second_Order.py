import numpy as np
import matplotlib.pyplot as plt


# How to solve dN/dt = -lamda*N
N0 = 1000
N = N0
lamda = 0.1
dt = 0.1
N_RK2 = []
time = []

for i in range(100):
    k1 = -lamda*N
    k2 = -lamda*(N+k1*dt/2)
    N += k2*dt
    N_RK2.append(N)
    time.append(i*dt)

time = np.array(time)
N_RK2 = np.array(N_RK2)
N_precise = N0*np.exp(-lamda*time)

plt.plot(time,N_RK2,'ro',label='Eular Algorithm')
plt.plot(time,N_precise,'k-',label='The precise graph of Nuclear Exposion')
plt.xlabel('Time')
plt.ylabel('Number of Nuclears')
plt.title('The solve of Nuclear_Eular')
plt.grid()
plt.show()
