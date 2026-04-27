import torch
import numpy as np
import matplotlib.pyplot as plt

#linear algebra
A = torch.arange(1,13).reshape(3,4)
print(A)
print(A*A)
B = torch.arange(1,25).reshape(4,6)
print(B@B.T)
print(A@B)
x = torch.arange(4)
print(A@x)


#calculus

#function
def f(x):
    return 3 * x ** 2 - 4 * x
x = np.linspace(-5,5, 100)
print(x.size)
y = f(x)
plt.figure(figsize=(20,8),dpi=200)
plt.title('f(x)-x',color='green')
plt.xlabel('x',color='red')
plt.ylabel('f(x)',color='blue')
plt.plot(x,y,label='f(x)',color='red',linestyle='-')

#deviditive
def Df(f,x): #求导
    epsilon = 1e-5
    return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
x0 = 0
def line(f,x0,x): #切线
    return Df(f,x0)*(x-x0)+f(x0)
z = line(f,x0,x)
plt.plot(x,z,label=f'line(x) at ({f(x0)},{x0})',color='blue',linestyle='--')
plt.show()

#intergrade
def F(f,a,b): #积分
    space = np.linspace(a,b,10000)
    eps = (b-a)/10000
    return sum([f(x)*eps for x in space])
print(F(f,0,5))