ax = {0:7,1:3,8:9,17:5}
bx = {0:8,5:3,8:3.5,10:6}
cx = {}
cxkeys = set(ax.keys())|set(bx.keys())
for k in cxkeys:
    cx[k] = ax.get(k,0)+bx.get(k,0)
lst = []
for i in sorted(cxkeys):
    lst.append(str(cx[i])+'x'+str(i))
polysum = '+'.join(lst)
print(polysum)