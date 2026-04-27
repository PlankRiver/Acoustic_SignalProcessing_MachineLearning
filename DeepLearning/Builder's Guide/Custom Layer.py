import torch
import torch.nn.functional as F
from torch import nn

#不带参数的图层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x-x.mean()
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
y = net(torch.rand(size=(4,8)))
print(y.mean())

#带参数的图层
class MyLinear(nn.Module):
    def __init__(self,in_unit,out_unit):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_unit,out_unit))
        self.bias = nn.Parameter(torch.randn(out_unit))
    def forward(self, x):
        linear = x@self.weight.data+self.bias.data
        return F.relu(linear)

dense = MyLinear(5,3)
print(dense.weight)
print(dense.weight.data)
print(dense.bias)
print(dense(torch.randn(2,5)))

net = nn.Sequential(MyLinear(64,8),MyLinear(8,1),CenteredLayer())
print(net(torch.rand(2,64)))
