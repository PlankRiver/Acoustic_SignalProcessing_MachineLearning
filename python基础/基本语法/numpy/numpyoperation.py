import numpy as np

# 随机抽样 np.random.choice(array,size,replace=True,p(权重))
# sample = ['apple','banana','cat']
# print(np.random.choice(sample,2,replace=False))
# print(np.random.choice(10,[3,3],replace=True,p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.2,0.05]))
# montecalor验证
# lst = ['apple','banana','cat','dog','elephant']
# percentage = [0.15,0.2,0.1,0.3,0.25]
# results = np.random.choice(lst,10000,p=percentage)
# value,key = np.unique(results,return_counts=True)
# for k,v in zip(value,key):
#     print(f'{k}:{v}次')

#np.unique(arr,return_index=False,return_inverse=False,return_count=False,axis)
#基本去重
# arr = np.array([1,3,3,2,3,4,2,1,2,4])
# print(np.unique(arr))
# arr = np.array(['apple','banana','apple','banana','cat','banana','cat'])
# print(np.unique(arr,return_index=True,return_counts=True))
# print(np.unique(arr,return_inverse=True))
# arr2 = [[1,2,3],[1,2,3],[4,5,6],[1,2,3],[4,5,6]]
# unique_rows = np.unique(arr2,axis=1)
# print(unique_rows)

#采样
# A = np.random.rand(6,3)
# mask = np.random.choice(np.arange(A.shape[0]),3,replace=True)
# print(A[mask])

#切片
# aArray = np.array([[1,2,3],[4,5,6]])
# print(aArray)
# print(aArray[::-1])
# print(aArray[:,::-1])
# print(aArray[[1,0],[0,2]])
# bArray = np.ones([7,7])
# bArray[1:-1,1:-1] = 0
# print(bArray)

#布尔索引
# array = np.arange(1,101)
# brray = array[array>50]
# print(brray)
# array[(array%2==0) | (array>50)] = -1
# print(array)

#np.where() 筛选与替换 np.where(condition,[x,y]) condition条件或布尔数组 x满足条件时取的值 y不满足条件时取的值
#只提供condition返回满足条件的索引元组 只提供xy返回与condition同形状的数组，元素来自xy
#仅传入条件
# array = np.array([3,1,4,1,5])
# indices = np.where(array>2)
# print(indices)
# print(array[indices])
#传入条件和替换值
# array = np.array([10,20,30,40,50])
# indices = np.where(array>25,'A','B')
# print(indices)
#多维数组处理
# matrix = np.array([[1,2],[3,4]])
# rows,cols = np.where(matrix>2)
# print(rows,cols,sep='\n')
#复杂条件组合
# array = np.array([5,-2,0,8,-1])
# results = np.where((array>0)&(array%2==0),'正偶',np.where(array<0,'负','其他'))
# print(results)
#数据清洗
# data = np.array([1,2,3,999,4,5,999])
# result = np.where(data==999,np.nan,data)
# print(result)
#阈值分类
# data = np.array([60,70,65,70,90,95,85,76,87,98,100,30,44])
# result = np.where(data>=90,'A',np.where(data>=80,'B',np.where(data>=70,'C',np.where(data>=60,'D','E'))))
# print(result)

#np.nan
#np.nan是非数字，属于浮点类型，不可进行比较，只能用np.isnan()
#创建含nan数组
# arr = np.array([1,np.nan,3,np.nan])
# print(arr)
# mask = np.isnan(arr)
# print(mask)
#统计时忽略nan
# print(np.nansum(arr))
# print(np.nanmean(arr))
#替换nan
# arr[np.isnan(arr)] = 0
# print(arr)
#过滤nan
# arr[arr==0] = np.nan
# arr = arr[~np.isnan(arr)]
# print(arr)
#处理缺失数据
# brr = np.array([1,2,np.nan,4,5,np.nan,6])
# print(np.nanmean(brr))
#图像处理
# photo = np.random.rand(5,5)
# photo[photo<0.1] = np.nan
# print(photo)

#ndarrya改变数组情况
#reshape
# arr = np.array([[1,2,3],[4,5,6]])
# size = arr.shape
# barray = arr.reshape(arr.shape[1],arr.shape[0])
# print(barray)
# print(np.arange(1,25).reshape(4,6))
#resize
# arr.resize(3,2)
# print(arr)
#np.transpose(arr,axes)  a.T简写 返回值是视图而非拷贝，共用一个内存，修改后值会影响前值
# a = np.array([[1,2,3],[4,5,6]])
# b = a.transpose()
# print(a)
# print()
# print(b)
# print()
# print(a.T)
# arr = np.arange(1,25).reshape(2,3,4)
# print(arr,'\n')
# brr = arr.transpose(0,2,1)
# print(brr)
#flatten为拷贝不共享内存 ravel为视图共享内存
# matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# arr = matrix.flatten(order='C')
# print(arr)
# arr = matrix.flatten(order='F')
# print(arr)

#拼接与分裂
#np.hstack()=np.concatenate((a,b),axis=1)与np.vstack()=np.concatenate((a,b),axis=0)
# barray = np.array([1,2,3])
# carray = np.array([4,5,6])
# print(np.hstack((barray,carray)))
# print(np.vstack((barray,carray)))
#np.split(arr,indices_or_sections,axis=0)
# arr = np.arange(9).reshape(3,3)
# new_arr = np.split(arr,3,axis=1)
# print(new_arr)
# print(np.split(arr,3,axis=0))
# print(np.split(arr,[1],axis=1))  #[0,1) and [1,3)
#np.hsplit()相当于np.split(axis=1)
#np.vsplit()相当于np.split(axis=0)

#基本运算
# aarray = np.array([2,2,2])
# barray = np.array([5,5,5])
# carray = aarray*barray
# darray = aarray+barray
# print(carray,darray,sep='\n')
#np.dot()
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# result = np.dot(a,b)
# print(result)
# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(np.dot(a,b))
# b = np.array([9,10])
# print(np.dot(a,b))
#np.dot(a,b)等价于a@b

#广播功能 较小的数组会被扩展到较大的数组上使之兼容
# a = np.array([[1,2,3],[4,5,6]])
# b = np.array([1,2,3])
# print(a+b,a*b,a-b,sep='\n')

#np.mean(axis,dtype,out,keepdims=False)
# arr = np.array([[1,2,3],[4,5,6]])
# print(arr-arr.mean(axis=1,keepdims=True))
# a = np.array([1,0,0,1,1,0,1,1,0])
# b = np.array([1,1,0,1,0,1,1,0,1])
# print((a==b))
# print(np.average(a==b))  #正确率

#np.sum()
# arr = np.array([[1,2,3],[4,5,6]])
# print(arr.sum())
# print(arr.sum(axis=0))
# print(arr.sum(axis=1))
# print(arr[arr>3].sum())
# arr.max()返回值 arr.argmax()返回索引

#average()用于加权
# arr = np.array([1,2,3,4,5])
# wght = np.array([3,6,7,9,2])
# print(np.average(arr,weights=wght))

#arr.argsort(axis=1)
# aarray = np.array([4,5,3])
# print(aarray.argsort())
# print(aarray[aarray.argsort()])
# barray = np.array([[3,4,2],[8,7,6]])
# print(barray.argsort())
# print(barray.argsort(axis=0))

#np.bincount(x) 统计非负元素出现的次数
# x = np.array([1,0,3,5,2,9])
# print(np.bincount(x))
#频率分布
# result = np.bincount(x)/len(x)
# print(result)