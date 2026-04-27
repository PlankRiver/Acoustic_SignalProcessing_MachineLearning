import numpy as np

#读入
readfile = np.loadtxt('ai_assistant_usage_student_life.csv',delimiter=',',dtype=str,skiprows=1,usecols=None)  #,unpack=True)
print(readfile)
print('*'*10)
print(readfile[[0,1,2,3,4],[0,0,1,1,1]])
print(readfile.shape)