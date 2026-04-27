import numpy as np
def primesearch(n):
    lst = np.arange(3,n+1,2)
    prime = [2]
    while len(lst)>0:
        a = lst[0]
        lst = lst[lst%a!=0]
        prime.append(int(a))
    return prime
print(primesearch(200))