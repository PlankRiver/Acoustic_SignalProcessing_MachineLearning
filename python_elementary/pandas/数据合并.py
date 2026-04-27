import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#数据合并
# t1 = pd.DataFrame(np.ones([2,4]),index=list('ab'),columns=list('xyzw'))
# t2 = pd.DataFrame(np.zeros([3,3]),index=list('abc'),columns=list('jkl'))
# print(t1)
# print(t2)
#join
# print(t1.join(t2))
# print(t2.join(t1))
#merge
# print(t1.merge(t2,left_on='x',right_on='k',how='inner')) #交集
# print(t1.merge(t2,left_on='x',right_on='k',how='outer')) #并集
# print(t1.merge(t2,left_on='x',right_on='k',how='left')) #左边为准
# print(t1.merge(t2,left_on='x',right_on='k',how='right')) #右边为准

#分类聚合
file = pd.read_csv('movies.csv')
# print(file.info())
# df = file.groupby('country')
# print(df)
# for i,j in df:
#     print(i)
#     print('-'*30)
#     print(j,type(j))
#     print('*'*30)
# #调用聚合方法
# print(df['genre'].count())
# print('*'*30)
# print(df.count())
# country_count = df['country'].count()
# print(country_count['China']>country_count['United States'])  #呜呜呜输了

#国内
chn = file[file['country']=='China']
print(file.info())
print(chn.info())
#多条件聚合
df = file[['name']].groupby(by=[file['director'],file['country']])
print(df.count())