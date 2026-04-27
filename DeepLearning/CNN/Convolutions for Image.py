import torch
from torch import nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y
X = torch.arange(9).reshape(3,3)
K = torch.arange(4).reshape(2,2)
Y = corr2d(X,K)
print(Y)

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size,kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return corr2d(x,self.weight)+self.bias

#边缘检测
X = torch.ones(6,8)
X[:,2:6] = 0
print(X)
K = torch.tensor([[1,-1]])
Y = corr2d(X,K)
print(Y)
print(corr2d(X.T,K))  #不能检测横向

#学习由X生成Y的卷积核
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X = X.reshape(1,1,6,8)
Y = Y.reshape(1,1,6,7)
for i in range(30):
    Y_hat = conv2d(X)
    l = (Y-Y_hat)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2*conv2d.weight.grad
    if (i+1)%2==0:
        print(f'batch{i+1}, loss{l.sum():.8f}')
print(conv2d(X))
print(conv2d.weight.data.reshape(1,2))