import torch
from torch import nn

def pool2d(X,pool_size,mode='max'):
    p_h,p_w = pool_size
    Y = torch.zeros(X.shape[0]-p_h+1,X.shape[1]-p_w+1)
    for i in range(p_h):
        for j in range(p_w):
            if mode=='max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode=='average':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(pool2d(X,(2,2),mode='max'))
print(pool2d(X,(2,2),mode='average'))

X = torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
print(X)

#深度学习框架步幅默认与池化层窗口大小相同
pool2d1 = nn.MaxPool2d(3)
print(pool2d1(X))
#自行调参
pool2d2 = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d2(X))
pool2d3 = nn.MaxPool2d((2,3),padding=(1,1),stride=(2,3))
print(pool2d3(X))

#池化层在每个通道上单独运算
X = torch.cat((X,X+1),1)
print(X)
pool2d4 = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d4(X))