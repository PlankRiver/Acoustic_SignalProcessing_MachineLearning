from scipy.special import roots_legendre
import matplotlib.pyplot as plt

def f(x):
    return x**4-2*x+1

N = 100
a,b = 0,2
x,w = roots_legendre(N)
# print(x,w)
# xp = x*(b-a)/2+(b+a)/2
# wp = w*(b-a)/2
#
# s=0
# for i in range(N):
#     s+=wp[i]*f(xp[i])
# print(s)

plt.plot(x,w)
plt.show()