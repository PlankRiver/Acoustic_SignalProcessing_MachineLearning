n = int(input('layers=:'))
lstpre = [1]
print(lstpre)
for i in range(n-1):
    lstpre.insert(0,0)
    lstpre.append(0)
    lstnew = []
    for j in range(len(lstpre)-1):
        lstnew.append(lstpre[j]+lstpre[j+1])
    print(lstnew)
    lstpre = lstnew