#Fibonaci-1
'''
f = [0]*20
f[0],f[1] = 0,1
for i in range(2,20):
    f[i] = f[i-1]+f[i-2]
print(f)
'''
#Fibonaci-2
n = 20
a,b = 0,1
i=0
while(i<n):
    print(b,end=' ')
    a,b = b,a+b
    i+=1