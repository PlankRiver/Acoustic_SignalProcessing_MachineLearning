#map,filter,reduce
#map(function,iterable,...)
a = [1,2,3,4,5,6]
b = list(map(lambda x:x**2,a))
print(b)
b = list(map(lambda x,y:f'{x}:{y}',a,b))
print(b)
a = ['apple','banana','cat','dophin','elephant']
print(list(map(len,a)))


dscore = {'a':[1,2,3],'b':[4,5,6],'c':[7,8,9],'d':[10,11,12],'e':[13,14,15]}
a = sorted(dscore.items(),key=lambda d:(d[0],d[1]),reverse=True)
# stra2 = dict(sorted(countstr(s).items(),key = lambda item:(item[1],item[0])))  #设置副条件
print(a)

#filter
#filter(function:True or False,iterable)
a = [1,2,3,4,5,6,7,8,9,10]
print(list(filter(lambda x:x%2==0,a)))
a = ['apple','banana','cat','dog','elephant']
print(list(filter(lambda x:len(x)>5,a)))

#reduce
from functools import reduce
#reduce(function,iterable) continuous calculation
a = [1,2,3,4,5,6,7,8,9,10]
b = reduce(lambda x,y:x+y,a)
print(b)
b = reduce(lambda x,y:x*y,a)
c = reduce(lambda x,y:x*y,a,1000)
print(b,c)
words = ['Python','is','awesome']
ans = reduce(lambda x,y:x+' '+y,words)
print(ans)