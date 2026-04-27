#string-function
#A-number
'''
def strcnt(x):
    s = f'{x:b}'
    return s.count('1')>s.count('0')
a,b = input('Please input two numbers').split()
a = int(a)
b = int(b)
for i in range(a,b+1):
    if strcnt(i):
        print(i)
'''
#字符串中的整数
'''
s = input('Please input a string')
def sintoint(s):
    w = ''
    neg = ''
    for x in s:
        if x.isdigit():
            if neg=='-':
                w+=neg
                neg=''
            w+=x
        else:
            w+=' '
            neg = x
    return w.split()
print(sintoint(s))
'''
#完全素数
def isprime1(n):
    for i in range(2,n):
        if i*i>n:
            break
        if n%i==0:
            return False
    return True
def isprime2(n):
    if set(str(n))<=set('2357'):
        return isprime1(n)
    return False
n = int(input())
print(isprime1(n),isprime2(n))