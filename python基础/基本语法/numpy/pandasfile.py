import pandas as pd
import numpy as np
#Series
# aser = pd.Series([1,2,5])  #有序字典
# print(aser)
# bser = pd.Series(['apple','banana','cat'],index = [1,2,3])
# print(bser)
# print(bser[3])
# print(bser*2)
# print(np.exp(aser))               #result:3    cat
# print(bser[2:3])                  #       dtype: object

#Dataframe
# data = {'name':['apple','banana','cat'],'pay':[4500,6000,8000]}
# df = pd.DataFrame(data)
# print(df)
# data = np.array([['apple',4500],['banana',6000],['cat',8000]])
# df = pd.DataFrame(data,index=np.arange(1,4),columns=['name','pay'])
# print(df)
# df = pd.DataFrame(data,columns=['name','pay']).set_index('name')  #set_index()将列转化为索引
# print(df)

#rows and columns
# data = {'name':['ab','cd','ef'],'pay':[4500,6000,8000]}
# df = pd.DataFrame(data)
#添加列添加行
# df['tax'] = [45,69,80]
# print('\n',df,sep='')
# df.loc[5] = ['gh',7500,75]  #df.loc[5] = {'name':'gh','pay':7500,'tax':75}
# print('\n',df,sep='')
# bf = pd.DataFrame({'name':['ert','yty'],'pay':[3000,9000],'tax':[30,90]},index=[7,9])
# print('\n',bf,sep='')
# # df.append(bf)
# # print('\n',df,sep='')    适用于旧版本
# pieces = [df,bf]
# print(pd.concat(pieces))

#删除
# data = {'name':['ab','cd','ef'],'pay':[4500,6000,8000]}
# df = pd.DataFrame(data)
# df['tax'] = [45,69,80]
# df.loc[5] = ['gh',7500,75]  #df.loc[5] = {'name':'gh','pay':7500,'tax':75}
# print(df.drop(5,axis=0))
# print(df.drop('tax',axis=1))
# df['tax'] = 30
# print(df)
# df.loc[5] = ['gh',9500,95]
# print(df)

#交换
# df = pd.DataFrame(np.arange(1,10).reshape(3,3),index=['F1','F2','F3'],columns=['A','B','C'])
# print(df)
# ind = ['F3','F1','F2']
# col = ['C','A','B']
# print(df.reindex(index=ind))   #df.reindex(ind,axis=0)
# print(df.reindex(columns=col))    #df.reindex(col,axis=1)

#未完待续





#数据选择  成绩
# data = [['zy',100,100,100,100],['cyy',98,67,80,62],['zxt',79,80,94,67],['zlf',62,100,76,63],['dyx',70,98,85,100]]
# df = pd.DataFrame(data,index=['a','b','c','d','e'],columns=['name','chinese','math','english','physics'])
# print(df,'\n',sep='')
# print(df['a':'c'],'\n',sep='')
# print(df[0:3],'\n',sep='')
# print(df.head(3),'\n',sep='')
# print(df.name,'\n',sep='')
# print(df[['name','chinese','physics']],'\n',sep='')
# #选择区域
# print(df.loc['b':'d','chinese':'english'],'\n',sep='')  #loc标签
# print(df.iloc[1:4,1:4],'\n',sep='')   #iloc位置
# print(df.loc['a':'c'])
# print(df.loc[:,'chinese':'english'])
# print(df.iloc[:,[1,2,3]])
# print(df.at['b','math'])   #at标签
# print(df.iat[1,2])    #iat位置
# print(df.iloc[:,[1,2,4,3]])   #与df.reindex(index=[...],column=[...])同
# #筛选
# df2 = df.loc['a':'c']
# temp = df.loc[df2.index]
# print(temp,'\n',sep='')
# print(df.math)
# print(df[(df.math>=80) & (df.index>='b') & (df.index<'e')])
# #查找
# print(df[df.name.isin(['zy','zxt'])])
# print(df[df.name.str.contains('^[z]')])

#数据统计与分析
#排序
# data = np.array([['zy',100,100,100,100],['cyy',98,67,80,62],['zxt',79,80,94,67],['zlf',62,100,76,63],['dyx',70,98,85,100]])
# df = pd.DataFrame(data,index=['a','b','c','d','e'],columns=['name','chinese','math','english','physics'])
# print(df,'\n',sep='')
# print(df.sort_values(by='chinese'))  #???100成了00？？
# print(df.sort_values(by='math')[:3].name)
#简单统计
# print(df.math.astype(int).mean())  #???
# print(df.info)
# print(df.describe())
#eval
df = pd.DataFrame([[1,2],[3,4],[5,6]],columns=['A','B'])
print(df,'\n',sep='')
print(df.eval('C=A+B'))
print(df.eval('C=sin(A)'),'\n',sep='')
#query
print(df.query('A==3'),'\n',sep='')
print(df.query('A+B>5'),'\n',sep='')
#value_counts()
data = [['zy',100,100,100,100],['cyy',98,67,80,62],['zxt',79,80,94,67],['zlf',62,100,76,63],['dyx',70,98,85,100]]
df = pd.DataFrame(data,index=['a','b','c','d','e'],columns=['name','chinese','math','english','physics'])
print(df.math.value_counts(),'\n',sep='')
df = df.eval('sum=chinese+math+english+physics')
print(df,'\n',sep='')
mark = ['A' if item>=320 else 'B' for item in df['sum']]
df['rank'] = mark
print(df,'\n',sep='')
print(df.groupby('rank').name.count(),'\n',sep='')
print(df.groupby(['name','rank'])['name'].count())
print(df.groupby(mark).groups)
for k,v in df.groupby(mark):
    print(k)
    print(v)
print()
print(df.groupby('rank')[['chinese','english']].mean())
#agg
print(df.groupby('rank')[['chinese','english']].agg(['mean','max']))
print(df.groupby('rank')[['chinese','english']].agg({'chinese':['mean','max'],'english':'mean'}))
def ranging(x):
    return x.max() - x.min()
print(df.groupby('rank')[['chinese','english']].agg(['mean','max',ranging]))
