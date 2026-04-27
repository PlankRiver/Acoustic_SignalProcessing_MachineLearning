import numpy as np
num = np.arange(1,101)
five = np.sqrt(5)
phi = (1+five)/2
fib = (phi**num-(-1/phi)**num)/five
fab = fib.astype(int)
print(fab)