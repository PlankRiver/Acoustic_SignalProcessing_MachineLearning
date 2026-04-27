import torch
from torch import nn
import torch.nn.functional as F

# x = torch.arange(24).reshape(4,6)
# torch.save(x,'x-file.txt')
# y = torch.load('x-file.txt')
# print(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
    def forward(self, x):
        return self.net(x)
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

x = torch.randn(2,20)
net = MLP()
net.apply(xavier)
y = net(x)
print(y)

torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
print(clone.state_dict())
print(clone(x)==net(x))