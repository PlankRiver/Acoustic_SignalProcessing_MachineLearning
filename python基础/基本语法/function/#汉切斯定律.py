#汉切斯定律
n = int(input())
def nicho(n):
    for i in range(1,n**3+1,2):
        sum = i
        l=i
        r=i
        while(sum<n**3):
            r+=2
            sum+=r
        if sum==n**3:
            return l,r
for num in range(1,n+1):
    l,r = nicho(num)[0],nicho(num)[1]
    s = [str(x) for x in range(l,r+1,2)]
    s = '+'.join(s)
    print(f'{num}^3='+s)