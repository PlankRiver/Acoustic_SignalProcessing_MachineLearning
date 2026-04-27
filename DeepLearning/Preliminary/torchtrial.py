import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.arange(12,dtype=torch.float32).reshape(3,-1)
x.to(device)
print(x)
print(x.shape)
print(x.numel)
print()
print(torch.zeros((2,3,4)))
print(torch.randn((2,3,4)))
print(x.exp())
y = torch.tensor([[1,2,3,4],[2,3,4,1],[3,4,1,2]])
print(y)
print(x+y,x*y)
print(torch.cat((x,y),dim=0),torch.cat((x,y),dim=1),sep='\n')
print(x==y)
print(x>y)