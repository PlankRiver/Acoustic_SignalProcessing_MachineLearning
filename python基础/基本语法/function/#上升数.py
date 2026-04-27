#上升数
n = input()
n = '0'+n+'0'
temp = ''
lst = []
for i in range(1,len(n)-1):
    if int(n[i])<int(n[i+1]):
        temp+=n[i]
    if int(n[i])>int(n[i+1]) and int(n[i])>int(n[i-1]) and i!=1:
        temp+=n[i]
        lst.append(temp)
        temp=''
if lst==[]:
    print('not found')
else:
    s = [int(x) for x in lst]
    for i in s[:-1]:
        print(i,end=',')
    print(s[-1])