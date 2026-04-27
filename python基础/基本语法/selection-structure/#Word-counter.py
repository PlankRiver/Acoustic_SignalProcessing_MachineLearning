#Word-counter
s = input('Enter an English sentence:')
s = s.lower()
slist = s.split()
sdict = {}
for item in slist:
    if item[-1] in ',.\'?!''':
        item = item[:-1]
    if item not in sdict:
        sdict[item] = 1
    else:
        sdict[item] += 1
print(sdict)