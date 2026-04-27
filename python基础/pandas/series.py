import matplotlib.pyplot as plt
import pandas as pd

#series是带标签的数组
t1 = pd.Series([1,4,6,2,3,9])
print(t1)
print('*'*20)
t2 = pd.Series([1,4,6,2,3,9],index=list('abcdef'))
print(t2)
dick = {'apple':45,'bat':50,'cat':30,'dick':'short','pussy':'big'}
t3 = pd.Series(dick)
print(t3)
print(t2.astype(float))
#数组中的索引同样可用
print(t3.index)
print(list(t3.index))
print(t3.values)
print(list(t3.values))