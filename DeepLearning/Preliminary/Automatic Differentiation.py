import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.arange(4.0)
x.requires_grad_(True)
y = 2*torch.dot(x,x)
print(y)
y.backward()
print(x.grad)
print(x.grad==4*x)

x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x*x
y.backward(gradient=torch.ones(len(y)))
print(x.grad)

x.grad.zero_()
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
print(x.grad)

x.grad.zero_()
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
d.backward()
print(a.grad)

#用自动求导来绘制图形
x_numpy = np.linspace(-10, 10, 1000)
x = torch.from_numpy(x_numpy).float().requires_grad_(True)
y = torch.sin(x)
y.sum().backward()
x_plot = x.detach().numpy()
y_plot = y.detach().numpy()
grad_plot = x.grad.detach().numpy()
#numpy<->torch自动求导绘图
plt.title('backward',color='green')
plt.xlabel('x',color='red')
plt.ylabel('y',color='blue')
plt.plot(x_plot,y_plot,label='sin(x)',color='red',linestyle='--',linewidth=2)
plt.plot(x_plot,grad_plot,label='sin\'(x)',color='blue',linestyle='-',linewidth=3) #-实线 --虚线 -.点划线 ：点虚线
plt.legend(loc='upper left')
plt.show()