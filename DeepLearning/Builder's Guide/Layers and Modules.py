import torch
import torch.nn as nn
import torch.nn.functional as F

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.output = nn.Linear(256, 10)
#     def forward(self, x):
#         return self.output(F.relu(self.hidden(x)))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,256),
                                 nn.ReLU(),nn.Linear(256,10))
    def forward(self, x):
        return self.net(x)

model = MLP()
x = torch.randn(2, 20, requires_grad=True)
# net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(x)
print(model(x))
print(model(x).sum())
print(model(x).shape)


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 使用 enumerate 获取索引 idx 和 层 block
        for idx, block in enumerate(args):
            # 关键修改：Key 必须是字符串！
            # 我们用 str(idx) 把 0, 1, 2 变成 "0", "1", "2"
            self._modules[str(idx)] = block

    def forward(self, x):
        # 这里的 .values() 获取的就是存进去的 block 对象
        for block in self._modules.values():
            x = block(x)
        return x
# 测试
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(x))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand(20,20)
        self.linear = nn.LazyLinear(20)
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x@self.rand_weight+1)
        x = self.linear(x)
        while x.abs().sum()>1:
            x/=2
        return x.sum()
net = FixedHiddenMLP()
print(net(x))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU(),)
        self.linear = nn.LazyLinear(16)
    def forward(self, x):
        return self.linear(self.net(x))
net = nn.Sequential(NestMLP(),nn.LazyLinear(20),FixedHiddenMLP())
print(net(x))

