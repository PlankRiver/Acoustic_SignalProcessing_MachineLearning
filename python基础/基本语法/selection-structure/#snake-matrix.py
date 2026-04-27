#snake-matrix
n = int(input())
if n>10:
    exit()
arr = []
for i in range(n):
    arr.append([0]*n)
k = 1
for i in range(n):
    if i%2==0:
        for j in range(n):
            arr[i][j] = k
            k+=1
    else:
        for j in range(n-1,-1,-1):
            arr[i][j] = k
            k+=1
for i in range(n):
    for j in range(n):
        print(f'{arr[i][j]:3d}',end='')
    print()