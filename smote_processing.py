import pandas as pd

df = pd.read_csv("df_smote_cv.csv")

print(df.head())

print(df.columns)

print(df.loc[:,'1'].unique())
print(df.loc[:,'1'].value_counts())