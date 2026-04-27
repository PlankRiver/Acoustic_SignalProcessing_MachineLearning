import numpy as np
import torch

a = np.linspace(0,10,100)

print(a*2)
print()

print(torch.cuda.is_available())
