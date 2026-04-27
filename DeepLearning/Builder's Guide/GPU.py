import torch
from torch import nn

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device('cuda'))
print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
print(try_gpu())
print(try_gpu(1))
print(try_all_gpus())

x = torch.tensor([1,2,3],dtype=torch.float)
print(x.device)
print(x)
y = torch.tensor([1,2,3],device=try_gpu())
print(y)
z = x.cuda(0)
print(z)
print(y+z)
try:
    print(x+y)
except:
    print('You count do calculations on different devices')

net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())
print(net(z))
print(net[0].weight.data.device) #net是个列表对象没有weight