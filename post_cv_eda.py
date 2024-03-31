import pandas as pd
import matplotlib.pyplot as plt

df_d = pd.read_csv("D_base_scores.csv",index_col=0)
df_d.plot(kind='box')
plt.title("Base-D model")
plt.show()

df_r = pd.read_csv("R_base_scores.csv",index_col=0)
df_r.plot(kind='box')
plt.title("Base-R model")
plt.show()