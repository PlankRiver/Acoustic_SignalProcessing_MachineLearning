import random
import torch
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt
import numpy as np

#tossing a coin
num = 100
head = sum([random.random()>0.5 for _ in range(num)])
print(head)

#sampling
fair_probs = torch.tensor([0.5,0.5])
print(Multinomial(1000,fair_probs).sample())

#plt
x = np.arange(1,1000,10)
y = [Multinomial(int(n),fair_probs).sample()/n for n in x]
y_head = [i[0].item() for i in y]
y_tail = [i[1].item() for i in y]
plt.plot(x,y_head,color='red',linestyle='--',label='head')
plt.plot(x,y_tail,color='blue',linestyle='-',label='tail')

plt.show()