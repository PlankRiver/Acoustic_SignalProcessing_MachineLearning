#标签编码
from sklearn.preprocessing import LabelEncoder

data = ['小学', '高中', '初中', '高中', '小学']
le = LabelEncoder()
encoded_data = le.fit_transform(data)
print(f"标签编码结果: {encoded_data}")
# 输出: [0 2 1 2 0]


#独热编码
import pandas as pd

df = pd.DataFrame({'城市': ['北京', '上海', '广州', '北京']})
# 使用 pandas 的 get_dummies 极其方便
one_hot = pd.get_dummies(df['城市'])
print(one_hot)