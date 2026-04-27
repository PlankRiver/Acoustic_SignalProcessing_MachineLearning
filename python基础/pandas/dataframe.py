import pandas as pd
import numpy as np
#pandas二维Series容器

t1 = pd.DataFrame(np.arange(24).reshape((4,6)),index=list('abcd'),columns=list('xyzwuv'))
print(t1)

# dick = {'apple':[22,33,44],'banana':[1,2,3],'cat':[44,53,23]}
# t2 = pd.DataFrame(dick)
# print(t2)
# print(t2.index)
# print(t2.columns)
# print(t2.values)
# print(t2.shape)
# print(t2.dtypes)
# print('*'*30)
# print(t2.describe())
# print(t2.info())
# print(t2.head(1))
# print('*'*30)
# print(t2.tail(2))

#dog names
t3 = pd.read_csv('dogNames2.csv')
# print(t3)
# print(t3.describe())
# print(t3.head())
# print(t3.info())
#排序
# print('*'*30)
# print()
# df = t3.sort_values(by='Count_AnimalName',ascending=False)  #降序
# print(df.head())
#索引
# print(t3[:20])
# print(t3['Row_Labels'])
# print('*'*30)
# print()
# print(t1.loc['a':'c','x':'u'])
# print(t1.loc[['a','c'],['x','v']])
#bool索引
t4 = pd.read_csv('ai_assistant_usage_student_life.csv')
print(t4)
print(t3[(t3['Row_Labels'].str.len()>4)&(t3['Count_AnimalName']>700)])
#缺失数据处理
t1.loc[['a','c'],['x','w','v']] = np.nan
print(t1)
print(t1.isnull())
print(t1.notnull())
print(t1['x'].mean())
print(t1.fillna(t1.mean()))
print(t1[pd.notnull(t1['x'])])