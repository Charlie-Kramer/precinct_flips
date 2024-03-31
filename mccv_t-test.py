from scipy.stats import ttest_ind
import pandas as pd

d_base = pd.read_csv("D_base_scores.csv",index_col=0)

print(d_base.describe())

print(ttest_ind(d_base.rf,d_base.knn,equal_var=False))

r_base = pd.read_csv("R_base_scores.csv",index_col=0)

print(r_base.describe())

print(ttest_ind(r_base.nn,r_base.rf,equal_var=False))