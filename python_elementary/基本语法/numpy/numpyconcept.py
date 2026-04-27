import numpy as np

#matrix
#np.identity()只能生成方阵，而np.eye()可以生成多种矩阵。
#more precisely: np.eye(n,m,k)，k为对角线偏移量，k>0为向上偏移
# print(np.identity(4))
# print(np.eye(3))
# print(np.eye(3,4))
# print(np.eye(3,k=1))

#arange
#np.arange(start,stop,step)默认不包括结尾  np.linspace(start,stop,num,endpoint=True/False)默认包括结尾
# print(np.arange(1,10,3))
# print(np.linspace(1,10,20))
# print(np.linspace(1,10,20,endpoint=False))

#np.zeros
# print(np.zeros([2,2]))
# print(np.ones([2,2]))
# print(np.full([3,3],np.pi))

#np.random
#np.emtpy()快速预分配内存
# print(np.empty([3,4]))
#np.random.random()生成[0,1)之间的均匀分布的随机数
# print(np.random.random())
# print(np.random.random(5))
# print(np.random.random([4,5]))
# print(10+(20-10)*np.random.random(10)) #用于生成[a,b)之间的n个随机数 a+(b-a)*np.random.random(n)
#np.random.rand() 与np.random.random()相似，不需要加括号，直接写np.random.rand(d0,d1,d2,...,dn)
#np.random.normal()
#np.random.normal(loc=0.0,scale=1.0,size=_),loc是均值,scale是标准差,size是输出的大小,模拟的就是N(loc,scale^2)
# print(np.random.normal(loc=5,scale=2))
# print(np.random.normal(size=5))
# print(np.random.normal(loc=5,scale=2,size=[5,5]))
#np.random.randn(size)是标准简化版的normal，默认是N(0,1),不需要加括号直接写np.random.randn(d0,d1,d2,....,dn)
#np.random.uniform(low,high,size)生成[loew,high)范围内的均匀分布的随机数
# print(np.random.uniform()) #默认是[0.0,1.0)
# print(np.random.uniform(low=0,high=5,size=[5,5]))
#dice
# dice = np.floor(np.random.uniform(low=1,high=7,size=10)).astype(int)  #np.floor()是向下取整，返回值是浮点数。np.floor(3.5)=3.0
# print(dice)
#RBG
# color = np.random.uniform(low=0,high=255,size=3).astype(int)
# print(color)
#montecalor
# siz = 10000000
# sample = np.random.uniform(size=[siz,2])
# num = np.sum(sample[:,0]**2+sample[:,1]**2<1)
# print(4*num/siz)
#np.random.randint(low,high,size,dtype=int) [low,high)
# print(np.random.randint(10))
# print(np.random.randint(10, 20))
# print(np.random.randint(10, 20, [3,4]))
#dice
# dices = np.random.randint(1,7,10)
# print(dices)
#生成随即索引
# arr = np.array([3,4,2,6,4,5,9,7,9,8,2,7,6,2,1,7])
# index = np.random.randint(0,len(arr),size=5)
# print(arr[index])

#np.fromfunction(function,size,dtype) 作用对象:坐标索引
# print(np.fromfunction(lambda i,j:(i+j)/2,[10,10],dtype=int))

#文件读写
# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# np.savetxt('nparray.txt',arr,fmt='%d',delimiter=',')
# asdf = np.loadtxt('nparray.txt',delimiter=',',dtype=int)
# print(asdf)
# data = np.loadtxt('nparray.txt',delimiter=',',dtype=float,skiprows=1,comments='#')
# print(data)