#函数式编程

#1
f = len
print(f([1,2,3,4,5,6]))
def calc(x,y,g):
    return g(x)+g(y)
print(calc('hello',[1,2,3],lambda x:len(x)**2))

#1
'''
f = eval(input())
print(calc(3,5,f))
f = eval(input())
print(f('abcd'))
'''

from functools import reduce
a,b,c = map(eval,input().split())
lst1 = list(map(lambda word:word.upper(),['nju','zju']))
lst2 = list(map(str.upper,['nju','zju']))
lst3 = ''.join(filter(str.isalpha,lst1))
lst4 = reduce(lambda x,y:x+y,range(1,101))