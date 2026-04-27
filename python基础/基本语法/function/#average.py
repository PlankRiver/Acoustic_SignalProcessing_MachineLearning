#average
def avg(d):
    min = 100
    max = 0
    for k,v in d.items():
        s = sum(v)/3
        if s>=max:
            max = s
            max_index = k
        if s<=min:
            min = s
            min_index = k
    return max_index,min_index
a = {'a':[1,2,3],'b':[4,5,6],'c':[7,8,9],'d':[10,11,12]}
x,y = avg(a)
print(x,y)