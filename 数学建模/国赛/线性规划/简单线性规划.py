import numpy as np
from scipy.optimize import linprog

#1.0
# 定义目标函数的系数（注意：由于是最大化，需要取负）
c = [-4, -3]   # 原目标函数：z = 4x + 3y → 最小化 -z = -4x -3y
# 不等式约束（左侧系数矩阵）
A = [[2, 1],   # 2x + y
     [1, 2]]   # x + 2y
# 不等式约束（右侧常数）
b = [20, 20]   # <=20, <=20
# 变量边界（x>=0, y>=0）
x_bounds = (0, None)
y_bounds = (0, None)
# 求解
res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')
# 输出结果
print('最优值:', -res.fun)  # 因为目标函数取负了，所以这里要取反
print('最优解:', res.x)
print(res)



#2.0
c = [2,3,1] #目标函数
#ub
A_ub = [[-3,-2,0]]
b_ub = [-6]
#eq
A_eq = [[1,4,2]]
b_eq = [3]
bounds = [(0,None),(0,None),(0,None)]
#solve
res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
print('<UNK>:', res.fun)
print('<UNK>:', res.x)


#3.0
c = [-2,-3,5]
A_ub = [[-2,5,-1],[1,3,1]]
b_ub = [-10,12]
A_eq = [[1,1,1]]
b_eq = [7]
bounds = [(0,None),(0,None),(0,None)]
res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
print('<UNK>:', -res.fun)
print('<UNK>:', res.x)

#4.0
c = [-3,1,1]
A_ub = [[1,-2,1],[4,-1,-2]]
b_ub = [11,-3]
A_eq = [[-2,0,1]]
b_eq = [1]
bounds = [(0,None),(0,None),(0,None)]
res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
print('<UNK>:', -res.fun)
print('<UNK>:', res.x)

#5.0 abs   max{|x1|+2|x2|+3|x3|+4|x4|}
c = [1,2,3,4,1,2,3,4]
A_eq = [[1,-1,-1,1,-1,1,1,-1],
        [1,-1,1,-3,-1,1,-1,3],
        [1,-1,-2,3,-1,1,2,-3]]
b_eq = [0,1,-1/2]
bounds = [(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(0,None)]
res = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
print('<UNK>:', res.fun)
print('<UNK>:', res.x)

#6.0 max:v
c = np.zeros([1,101])
c[-1] = -1
np.random.seed(0)
A = np.random.randint(0,11,size=[100,150])
A_ub = np.zeros([150,101])
for j in range(150):
     for i in range(100):
          A_ub[j,i] = -A[i,j]
     A_ub[j,-1] = 1
b_ub = np.zeros(150)
A_eq = np.zeros([1,101])
A_eq[0,:100] = 1
b_eq = [1]
bounds = [(0,None)]*100+[(None,None)]
res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
print('<UNK>:', -res.fun)
print('<UNK>:', res.x)