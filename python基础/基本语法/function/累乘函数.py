n = int(input())
f = eval(input())
def mult(n,f):
    ans = 1
    for i in range(1,n+1):
        ans*=f(i)
    return ans
print(mult(n,f))