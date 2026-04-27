#list-analysis
'''
from math import sqrt
n = int(input('input a number:'))
for i in range(2,int(sqrt(n))+1):
    if n%i==0:
        print('The number is not a prime.')
        break
else:
    print('The number is a prime')
'''

a = [x**2 for x in range(10)]
print(a)
b = [[0]*5 for i in range(5)]
print(b)
c = [x**2 for x in range(10) if x**2<50]
print(c)
d = [(x+1,y+1) for x in range(2) for y in range(2)]
print(d)
print()
e = [eval(x) for x in ['23','3.5']]
f = [str(x) for x in [23,3.5]]
g = [line[0] for line in [[1,2,3],[4,5,6],[7,8,9]]]
h = [chr(97+x) for x in range(26)]
print(e,f,g,h,sep='\n')
#二进制ip转换为标准ip形式
ip = '10101100000100001111111000000001'
IP = '.'.join([str(int(ip[8*i:8*(i+1)],2)) for i in range(4)])
print(IP)