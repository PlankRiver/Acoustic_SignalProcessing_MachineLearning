import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

alist = []
qlist = []
for i in range(1,51):
    alist.append(i*0.001)
for a in alist:
    c = [-0.05,-0.27,-0.19,-0.185,-0.185]
    A_eq = [[1,1.01,1.02,1.045,1.065]]
    b_eq = [1]
    A_ub = [[0,0,0,0,0],
            [0,0.025,0,0,0],
            [0,0,0.015,0,0],
            [0,0,0,0.055,0],
            [0,0,0,0,0.026]]
    b_ub = [a,a,a,a,a]
    bounds = [(0,2),(0,2),(0,2),(0,2),(0,2)]
    res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
    qlist.append(-res.fun)
plt.plot(alist,qlist)
plt.xlabel('Advance Index')
plt.ylabel('Profit')
plt.grid(True)
plt.show()