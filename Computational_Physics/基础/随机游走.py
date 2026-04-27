# # 初级版
# import random
# import matplotlib.pyplot as plt
#
# # Random Walk
# def random_walk(steps):
#     x = [0.0]
#     y = [0.0]
#     cur_x,cur_y = 0.0,0.0
#     for i in range(steps):
#         dx,dy = random.normalvariate(0,1), random.normalvariate(0,1)
#         cur_x,cur_y = cur_x + dx,cur_y + dy
#         x.append(cur_x)
#         y.append(cur_y)
#     return x,y
#
# # Paint
# x_paint,y_paint = random_walk(1000)
# plt.plot(x_paint,y_paint)
# plt.show()
#
# # R_quare~t
# steps = int(input("Enter the number of steps: "))
# num = int(input("Number of particles: "))
#
# R_squared_sum = [0.0 for _ in range(steps+1)]
# for i in range(num):
#     x,y = random_walk(steps)
#     for j in range(steps+1):
#         R_squared_sum[j] += x[j]**2+y[j]**2
# R_squared_avg = [val/num for val in R_squared_sum]
# T = [0.01*t for t in range(steps+1)]
#
# plt.plot(T,R_squared_avg)
# plt.show()
#
#
#
#
# # Numpy优化版
# import numpy as np
# import matplotlib.pyplot as plt
#
# steps = int(input('How many steps'))
# num = int(input('How many particles'))
# dt = 0.01
#
# x = np.random.normal(0,1,(num,steps))
# y = np.random.normal(0,1,(num,steps))
# T = np.arange(steps+1)*dt
#
# x = np.cumsum(x,axis=1)
# y = np.cumsum(y,axis=1)
# x = np.insert(x,0,0,axis=1)
# y = np.insert(y,0,0,axis=1)
#
# R_square = x**2+y**2
# R_square_avg = np.mean(R_square,axis=0)
# T = np.arange(steps+1)*dt
#
# plt.plot(T,R_square_avg)
# plt.show()
#
#
#
#
# import random
# import matplotlib.pyplot as plt
#
# def random_walker_generator(steps):
#     x,y = [0.0],[0.0]
#     cur_x,cur_y = 0.0,0.0
#     for i in range(steps):
#         cur_x += random.normalvariate(0,1)
#         cur_y += random.normalvariate(0,1)
#         x.append(cur_x)
#         y.append(cur_y)
#     return x,y
#
# def draw_random_walker_graph(steps):
#     x,y = random_walker_generator(steps)
#     plt.plot(x,y)
#     plt.show()
#
# def draw_random_walker_R_t(steps,num,dt):
#     R_square = [0.0 for i in range(steps+1)]
#     for i in range(num):
#         x,y = random_walker_generator(steps)
#         for j in range(len(x)):
#             R_square[j] += x[j]*x[j]+y[j]*y[j]
#     R_square_avg = [val/num for val in R_square]
#     T = [dt*i for i in range(steps+1)]
#     plt.plot(T,R_square_avg)
#     plt.show()
#
# if __name__ == '__main__':
#     steps = int(input('How many steps?'))
#     num = int(input('How many particles?'))
#     dt = float(input('How long per step?'))
#     draw_random_walker_graph(steps)
#     draw_random_walker_R_t(steps,num,dt)


import numpy as np
import matplotlib.pyplot as plt

def random_walker_generator(steps,num):
    x = np.random.normal(0,1,(num,steps))
    y = np.random.normal(0,1,(num,steps))

    x = np.cumsum(x,axis=1)
    y = np.cumsum(y,axis=1)
    x = np.insert(x,0,0,axis=1)
    y = np.insert(y,0,0,axis=1)

    return x,y

def draw_random_walker_graph(steps,num):
    x,y = random_walker_generator(steps,num)
    select = np.random.randint(0,num)
    x = x[select]
    y = y[select]
    plt.plot(x,y)
    plt.show()

def draw_random_walker_R_t(steps,num,dt):
    x,y = random_walker_generator(steps,num)
    R_square = x*x+y*y
    R_means = np.mean(R_square,axis=0)
    T = np.arange(steps+1)*dt
    plt.plot(T,R_means)
    plt.show()

if __name__ == '__main__':
    steps = int(input('How many steps?'))
    num = int(input('How many particles?'))
    dt = float(input('How long per step?'))
    draw_random_walker_graph(steps,num)
    draw_random_walker_R_t(steps,num,dt)