import numpy as np
import matplotlib.pyplot as plt

# How to solve dN/dt = -lamda*N
N0 = 1000
N = N0
lamda = 0.1
dt = 0.1
N_Eular = []
time = []

for i in range(100):
    N_Eular.append(N)
    time.append(i*dt)
    N = N - lamda*N*dt

time = np.array(time)
N_Eular = np.array(N_Eular)
N_precise = N0*np.exp(-lamda*time)

plt.plot(time,N_Eular,'ro',label='Eular Algorithm')
plt.plot(time,N_precise,'k-',label='The precise graph of Nuclear Exposion')
plt.xlabel('Time')
plt.ylabel('Number of Nuclears')
plt.title('The solve of Nuclear_Eular')
plt.grid()
plt.show()
