import numpy as np
from scipy.optimize import linprog

# 定义决策变量:
# [x1, x2, x3, x4, x5, x6, x7, x8, x9, u1, u2, u3, u4, u5]
# x1: 产品I在A1加工数量, x2: 产品I在A2加工数量
# x3: 产品II在A1加工数量, x4: 产品II在A2加工数量
# x5: 产品III在A2加工数量
# x6: 产品I在B1加工数量, x7: 产品I在B2加工数量, x8: 产品I在B3加工数量
# x9: 产品II在B1加工数量
# x10: 产品II在B2加工数量
# u1-u5: 设备使用率 (0-1之间)

A1_cost = 300
A2_cost = 321
B1_cost = 250
B2_cost = 783
B3_cost = 200

A1_capacity = 6000
A2_capacity = 10000
B1_capacity = 4000
B2_capacity = 7000
B3_capacity = 4000

cost = [0.25,0.35,0.50]
price = [1.25,2.00,2.80]
profit = [price[i]-cost[i] for i in range(len(cost))]

#linprog
c = [-profit[0],-profit[0],
     -profit[1],-profit[1],
     -profit[2],
     0,0,0,0,0,
     A1_cost,A2_cost,B1_cost,B2_cost,B3_cost]
A_ub = [[5,0,10,0,0,0,0,0,0,0,-6000,0,0,0,0],
        [0,7,0,9,12,0,0,0,0,0,0,-10000,0,0,0],
        [0,0,0,0,0,6,0,0,8,0,0,0,-4000,0,0],
        [0,0,0,0,0,0,1,0,0,1,0,0,0,-7000,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,-4000]]
b_ub = [0,0,0,0,0]
A_eq = [[1,1,0,0,0,-1,-1,-1,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,-1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0]]
b_eq = [0,0,0]
bounds = [(0,None)]*10+[(0,1)]*5

res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
x = res.x
print(f'最大收益:{-res.fun}')
print("\n生产计划:")
print(f"产品I在A1上加工: {x[0]:.0f}件")
print(f"产品I在A2上加工: {x[1]:.0f}件")
print(f"产品II在A1上加工: {x[2]:.0f}件")
print(f"产品II在A2上加工: {x[3]:.0f}件")
print(f"产品III在A2上加工: {x[4]:.0f}件")
print(f"产品I在B1上加工: {x[5]:.0f}件")
print(f"产品I在B2上加工: {x[6]:.0f}件")
print(f"产品I在B3上加工: {x[7]:.0f}件")
print(f"产品II在B1上加工: {abs(x[8]):.0f}件")
print(f"产品III在A2上加工: {x[9]:.0f}件")
print(f'利用率：{x[10:15]}')