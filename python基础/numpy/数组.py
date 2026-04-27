import numpy as np

#一维数组
t1 = np.array([1,2,3])
print(t1)
print(type(t1))
t2 = np.array(range(7))
print(t2)
t3 = np.arange(12)
print(t3)
t4 = np.arange(4,20,2)
print(t4)
print(t4.dtype)
t5 = np.array([1,2,3,4,5],dtype=float)
print(t5)
t6 = np.array([1,1,0,0,1,0,1,1,0,1],dtype=bool)
print(t6)
t7 = t5.astype(int)
print(t7)

#np.round()
# 对单个数值舍入
print(np.round(3.14159))      # 输出: 3.0
print(np.round(3.14159, 2))   # 输出: 3.14
# 对数组舍入
arr = np.array([1.234, 5.678, 9.012])
print(np.round(arr))          # 输出: [1. 6. 9.]
print(np.round(arr, 1))       # 输出: [1.2 5.7 9.0]

#多维数组
a = np.array([[1,2,3],[4,5,6]])
print(a,a.shape)
b = np.arange(12).reshape([3,4])
print(b)
print(a.reshape(6,))
print(a.flatten())
#加法广播
print(a+5)

#bool索引
a = np.ones([7,7])
a[1:-1,1:-1] = 8
b = np.where(a<6,0,100)  #三元运算符
print(b)
print('*'*10)
print(a.shape)  #Tuple

#转置
a = np.arange(24).reshape(4,6)
print(a)
print(a.T)  #转置

#交换 a[[1,2],:]=a[[2,1],:]行交换  a[:,[0,2]]=a[:,[2,0]]列交换

#copy and view
#a = b 在numpy中并非赋值 相互影响 应用a = b.copy()

#nan
a = np.array([1.,2.,5.,np.nan])
print(np.isnan(a))
print(np.nan!=np.nan)
print(np.count_nonzero(np.isnan(a)))
print(np.count_nonzero(a!=a))
b = np.arange(24).astype(float).reshape(4,6)
b[2,3] = np.nan

#常用统计方法
print('*'*10)
print(b)
print(np.sum(b))
print(np.sum(b,axis=0))
print(np.sum(b,axis=1))
print(np.mean(b,axis=0))
print(np.max(b,axis=0))
print(np.min(b,axis=0))
print(np.std(b,axis=0)) #标准差
print(np.ptp(b,axis=0)) #极差
