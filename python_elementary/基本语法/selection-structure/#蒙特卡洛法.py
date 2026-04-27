#蒙特卡洛法
import random
n = int(input('As big as you can imagine'))
lstx = [random.uniform(-1,1) for x in range(n)]
lsty = [random.uniform(-1,1) for y in range(n)]
coordinate = zip(lstx,lsty)
count = 0
for xy in coordinate:
    if xy[0]**2+xy[1]**2<=1:
        count+=1
pi = 4*count/n
print(pi)