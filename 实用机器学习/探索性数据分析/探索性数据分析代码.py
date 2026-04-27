import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data = pd.read_feather(Path(__file__).with_name('house_sales.ftr'))
print(data.shape)
print(data.head())

# We drop columns that at least 30% of the data are null to simplify our EDA.
null_num = data.isnull().sum()
print(null_num)
print(data.columns[null_num < 0.3*len(data)])
data.drop(columns=data.columns[null_num > 0.3*len(data)], inplace=True)

print(data.dtypes)

#转换为浮点数
currency = ['Sold Price','Listed Price','Tax assessed value','Annual tax amount']
for c in currency:
    data[c] = data[c].replace(r'[$,-]',regex=True).replace(r'^\s*$',np.nan, regex=True).astype(float)
areas = ['Total interior livable area','Lot size']
for c in areas:
    acres = data[c].str.contains('Acres')==True
    col = data[c].replace(r'\b sqft\b|\b Acres\b|\b,\b','',regex=True).astype(float)
    col[acres] *= 43560
    data[c] = col
