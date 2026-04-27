s = input()
ans = ''
for i in s:
    if i.isdigit() or i=='-':
        ans+=i
    else:
        ans+=' '
words = ans.split()
print(*words,sep=',')
