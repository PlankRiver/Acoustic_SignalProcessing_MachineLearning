#algebra
#gcd
'''
a = int(input('Please input a number'))
b = int(input('Please input a number'))
def gcd(x,y):
    while x%y!=0:
        x,y =  y,x%y
    return y
c = gcd(a,b)
print(c)
print(a*b//c)
'''
#isprime
def isprime(x):
    if x%2==0:
        return (x==2)
    else:
        for i in range(3,x,2):
            if i*i>x:
                break
            if x%i==0:
                return False
        return True
for x in range(2,100):
    if isprime(x):
        print(x,end=' ')
print()

date = {1:31,2:28,3:31}
for m in date.keys():
    for d in range(1,date[m]+1):
        day = 2025*10000+m*100+d
        if isprime(day):
            print(f'2025{m:0>2}{d:0>2}')