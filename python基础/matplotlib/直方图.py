import matplotlib.pyplot as plt
import random

a = [random.randint(15,40) for _ in range(100)]
plt.xticks(range(min(a),max(a)+1))
plt.hist(a,25)
plt.grid()
plt.show()