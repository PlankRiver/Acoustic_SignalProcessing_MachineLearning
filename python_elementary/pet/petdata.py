import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#读取并检验数据
data_file = Path(__file__).resolve().parent / 'aac_intakes_outcomes.csv'
df = pd.read_csv(data_file,
                 parse_dates=['outcome_datetime', 'intake_datetime', 'date_of_birth'])
print(f"Data shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

#数据加工（针对时间：将年月统一转化为日）
def convert_age_to_days(age_str):
    if pd.isna(age_str):
        return np.nan
    parts = age_str.split()
    num = float(parts[0])
    unit = parts[1]
    if 'year' in unit:
        return num * 365
    elif 'month' in unit:
        return num * 30
    elif 'week' in unit:
        return num * 7
    else:
        return num
df['age_intake_days'] = df['age_upon_intake'].apply(convert_age_to_days)
df['age_outcome_days'] = df['age_upon_outcome'].apply(convert_age_to_days)
df['shelter_days'] = (df['outcome_datetime'] - df['intake_datetime']).dt.days
#时间特征
df['outcome_month'] = df['outcome_datetime'].dt.month
df['outcome_weekday'] = df['outcome_datetime'].dt.weekday
df['intake_month'] = df['intake_datetime'].dt.month

# 数据分析与可视化

#动物类型分布
plt.figure(figsize=(10, 6))
df['animal_type'].value_counts().plot(kind='bar')
plt.title('Animal Type Distribution')
plt.xlabel('Animal Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#收容结果分析
plt.figure(figsize=(12, 6))
outcome_counts = df['outcome_type'].value_counts()
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%')
plt.title('Outcome Type Distribution')
plt.show()

#年龄分析
plt.figure(figsize=(12, 6))
df.boxplot(column='age_intake_days', by='animal_type')
plt.title('Age Distribution by Animal Type')
plt.suptitle('')
plt.ylabel('Age (days)')
plt.xlabel('Animal Type')
plt.xticks(rotation=45)
plt.show()

#动物收容数量分析
plt.figure(figsize=(12, 6))
# monthly_outcomes = df.resample('ME', on='outcome_datetime').size()
# monthly_outcomes.plot(label='Monthly Outcomes')
plt.title('Monthly Animal Outcomes')
plt.xlabel('Date')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#品种分析
top_breeds = df['breed'].value_counts().head(20)
plt.figure(figsize=(12, 8))
top_breeds.plot(kind='barh')
plt.title('Top 20 Breeds')
plt.xlabel('Count')
plt.ylabel('Breed')
plt.tight_layout()
plt.show()

#收容的时间分析
plt.figure(figsize=(12, 6))
df['shelter_days'].plot(kind='hist', bins=50, range=(0, 100))
plt.title('Distribution of Time in Shelter (Days)')
plt.xlabel('Days in Shelter')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#季节分析
plt.figure(figsize=(12, 6))
df['outcome_month'].value_counts().sort_index().plot(kind='bar')
plt.title('Outcomes by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()