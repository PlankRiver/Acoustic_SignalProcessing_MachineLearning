#random-trial
'''
import random
method = ['get','post','update','delete']
url1 = 'service'
url2 = 'controller'
url3 = 'action'
output = ['xx:xx.xxx '+random.choice(method)+' /'+url1+str(random.randint(1,3))+'/'+url2+str(random.randint(1,3))+str(random.randint(1,3))+'/'+url3+str(random.randint(1,3))+str(random.randint(1,3))+str(random.randint(1,3)) for _ in range(10)]
print(output)
'''
import random
def func(data):
    cls = random.choice(list(data.keys()))
    stu = random.randint(1,data[cls])
    return f'{cls}{stu:02}'
data = {'A001':32,'A002':47,'A003':35,'A004':49}
result = set()
while len(result)<10:
    result.add(func(data))
print(result)