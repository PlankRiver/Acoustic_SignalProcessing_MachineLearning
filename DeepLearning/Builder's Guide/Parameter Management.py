import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
x = torch.rand(size=(2,4))
y = net(x)
print(y)

#参数访问
print(net[2].state_dict())

#目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad==None)

#一次性访问所有参数
print(*[(name,parameter.shape) for name,parameter in net[0].named_parameters()])
print(*[(name,parameter.shape) for name,parameter in net.named_parameters()])
print(net.state_dict()['2.bias'].data)

#从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}',block1())
    return net
rgnet = nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet)
print(rgnet(x))

#内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data,net[0].bias.data)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data,net[0].bias.data)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data,net[2].weight.data)

#自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            'Init',
            *[(name,parameter.shape) for name,parameter in m.named_parameters()][0]
        )
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.abs()>0
net.apply(my_init)
print(net[0].weight.data,net[2].weight.data)
net[0].weight.data += 1
net[0].weight.data[0,0] = 42
print(net[0].weight.data)

#参数绑定
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
y = net(x)
print(net[2].weight.data==net[4].weight.data)
net[2].weight.data[0,0] = 42
print(net[2].weight.data==net[4].weight.data)
