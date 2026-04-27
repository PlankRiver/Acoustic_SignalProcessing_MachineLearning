import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file = pd.read_csv('ai_assistant_usage_student_life.csv')
# fig = file['SatisfactionRating'].values
# print(fig)
# max_fig = fig.max()
# min_fig = fig.min()
# num = int((max_fig - min_fig)/0.2)+1
#
# plt.figure(figsize=(20,8),dpi=80)
# plt.hist(fig,bins=num)
# plt.xticks(np.arange(min_fig, max_fig,0.2))
# plt.show()

#movie
file = pd.read_csv('movies.csv')
print(file.info())
print(file.head())
df_genre_count = file['genre']
print(df_genre_count)
print('*'*30)
print(file.shape[0])