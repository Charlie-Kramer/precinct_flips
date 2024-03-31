import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gp
import matplotlib.patches as mpatches
from scipy.stats import anderson_ksamp
from pysal.lib import weights
from pysal.explore import esda
import numpy as np

df = pd.read_csv("voting_acs_df.csv").dropna()
df["GEOID"] = df["GEOID"].astype(int)
flip_counts = df["flip4cat"].value_counts()
print(flip_counts)

plt.figure(figsize=(8,6))
plt.bar(flip_counts.index,flip_counts)
plt.xlabel("Flip")
plt.ylabel("Count")
plt.show()

df1 = df.loc[:,["flip4cat","Pop","PctMale","PctWhite","MedAge","PctForn","PctPoverty","PctBroadband","PctMedicaid"]]

dd = df1.describe()
print(df1.groupby(["flip4cat"]).mean())

#
# draw map
# need to get precinct geometries
#

precincts_2020 = gp.read_file("/Users/charleskramer/Documents/virginia-voting-precincts-master/maps/va-precinct-shapefiles-2019-2020/va-precincts-2019")

precincts_merged = precincts_2020.merge(df,left_on='precUid',right_on="PrecinctUid")
color_dict = {"DR":"pink","DD":"blue","RD":"purple","RR":"red"}
precincts_merged["color"] = precincts_merged["flip4cat"].map(color_dict)

precincts_merged.plot(color = precincts_merged.color)
red_patch = mpatches.Patch(color='red',label="R->R")
purple_patch = mpatches.Patch(color='purple',label="R->D")
pink_patch = mpatches.Patch(color='pink',label="D->R")
blue_patch = mpatches.Patch(color="blue",label="D->D")
plt.legend(handles=[red_patch, blue_patch, pink_patch, purple_patch])
plt.xticks([])
plt.yticks([])
plt.show()


#
# boxplots of vars grouped by 2020flip
#
# Identify numerical columns
sns.set_style("darkgrid")


numerical_columns = df1.select_dtypes(include=["int64", "float64"]).columns

# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    print("feature",feature)
    sns.histplot(df1.loc[:,feature], kde=True)
    #plt.title(f"{feature} | Skewness: {round(df1[feature].skew(), 2)}")

# Adjust layout and show plots
plt.tight_layout()
plt.show()

fig_dict = {"DR":"D->R","DD":"D->D","RD":"R->D","RR":"R->R"}
color_dict = {"DR":"pink","DD":"blue","RD":"purple","RR":"red"}
fig, axs = plt.subplots(len(numerical_columns),3)
for idx, feature in enumerate(numerical_columns, 1):
    print("feature",feature)
    for i in ["DD","DR","RD","RR"]:
        plt.subplot()
        sns.histplot(df1[df1["flip4cat"]==i].loc[:,feature], kde=True,label=fig_dict[i],color=color_dict[i])
        plt.title(f"{feature}" )
        plt.legend()
    plt.show()

#
# Anderson-darling tests for equality of distributions
#

test_vars = numerical_columns.drop("Pop")

for var in test_vars:
    test_cols = []
    for fliptype in ["DD","DR","RD","RR"]:
        test_cols.append(df1.loc[df1["flip4cat"]==fliptype,var].tolist())
    print("anderson",var,anderson_ksamp(test_cols))


precincts_merged["drflip"] = np.where(precincts_merged["flip4cat"]=="DR",1,0)
precincts_merged["rdflip"] = np.where(precincts_merged["flip4cat"]=="RD",1,0)
w = weights.KNN.from_dataframe(precincts_merged,k=8)
jc_dr = esda.join_counts.Join_Counts(precincts_merged["drflip"],w)
print("DR")
print("counts: bb, bw, ww",jc_dr.bb,jc_dr.bw,jc_dr.ww)
print("p vals: bb/bw",jc_dr.p_sim_bb,jc_dr.p_sim_bw)
jc_rd = esda.join_counts.Join_Counts(precincts_merged["rdflip"],w)
print("RD")
print("counts: bb, bw, ww",jc_rd.bb,jc_rd.bw,jc_rd.ww)
print("p vals: bb/bw",jc_rd.p_sim_bb,jc_rd.p_sim_bw)
stob = 1