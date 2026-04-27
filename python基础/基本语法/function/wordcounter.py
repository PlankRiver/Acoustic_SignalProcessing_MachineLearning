# def cnt(words):
#     d = {}
#     for word in words:
#         d[word] = d.get(word, 0) + 1
#     return d
# s = input().lower()
# for i in ['?','!',',','.','\"','\'','\\','<','>','@','#','%','(',')','*','&','$']:
#     s = s.replace(i,' ')
# d = dict(sorted(cnt(s.split()).items(),key=lambda x:(x[1],x[0]),reverse=True))
# for k,v in d.items():
#     print(f'{k}:{v}')

def cnt(words):
    d = {}
    for word in words:
        d[word]=d.get(word,0)+1
    return d
s = input()
for icon in [',','.','!','?']:
    if icon in s:
        s = s.replace(icon,' ')
s = s.lower().split()
dct = cnt(s)
dct2 = dict(sorted(dct.items(),key=lambda item:(item[1],item[0]),reverse=True))
for k,v in dct2.items():
    print(f'Word {k} has {v} times')