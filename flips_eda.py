# eda of the voting data
# counts up precinct flips
# writes it to file df0.csv

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#https://github.com/erikalopresti/virginia-voting-precincts precinct shapefiles

type_dict = {'CandidateUid': 'str', 'FirstName': 'str', 'MiddleName':'str', 'LastName':'str', 'Suffix':'str',
       'TOTAL_VOTES':'float', 'Party':'str', 'WriteInVote':'str', 'LocalityUid':'str', 'LocalityCode':'str',
       'LocalityName':'str', 'PrecinctUid':'str', 'PrecinctName':'str', 'DistrictUid':'str',
       'DistrictType':'str', 'DistrictName':'str', 'OfficeUid':'str', 'OfficeTitle':'str',
       'ElectionUid':'str', 'ElectionType':'str', 'ElectionDate':'str', 'ElectionName':'str'}

df0 = pd.DataFrame()
datadir = os.scandir("../Va General Election Results/")

for i,z in enumerate(datadir):
    print(z.name)
    if (z.name.endswith('.csv')):
        df = pd.read_csv("../Va General Election Results/"+z.name,dtype=type_dict)
        precinct_ids = []

        if "PrecinctUid" in df.columns:
            precinct_ids.append(set(df["PrecinctUid"].tolist()))
            print("in")
            year = [int(x[0:4]) for x in df["ElectionDate"]]
            df["Year"] = year
            df2 = pd.DataFrame(df.groupby(['ElectionDate','PrecinctUid','Party'])['TOTAL_VOTES'].sum())
            df2.reset_index(inplace=True,level=['Party'])
            df3 = df2.pivot(columns='Party')
            df3["Blue"] = np.where(df3[('TOTAL_VOTES','Democratic')]>df3[('TOTAL_VOTES','Republican')],1,0)

            #df3.reset_index(inplace=True,level=['ElectionDate','PrecinctUid'])
            df3.reset_index(inplace=True, level=['PrecinctUid'])
            df4 = df3.loc[:,["PrecinctUid","Blue"]]
            df5 = df4.pivot(columns="PrecinctUid")
            df5["Year"] = year[0]
            cols = [c[-1] for c in df5.columns.tolist()]
            cols_dict = dict(zip(df5.columns.tolist(),cols))
            df6 = df5.rename(columns=cols_dict)
            df0 = pd.concat([df0,df6])

    else:
        print("skip")


df0 = df0.sort_values(by='Year')
df0 = df0.loc[:,(df0.isna().sum(axis=0)<=16)]
m = df0.iloc[:,:].max() - df0.iloc[:,:].min()
m.sort_values(inplace=True,ascending=False)
print(m.iloc[:].value_counts())
buh = 1
cols0 = df0.columns.tolist()
cols = [c[2] for c in cols0]
coldict = dict(zip(cols0,cols))
df0.columns = cols
df0_T = df0.transpose()
df0_T.to_csv('df0.csv')

test_results = df0.reset_index(drop=True).drop(axis=1,labels=[''])

c = [[0,0],[0,0]] # totals: index 0 = red, index 1 = blue 00 = red, red, [1, 0] = red,blue etc, row = 2019 col = 2020
c_ts = [] # time series
for i in range(1,len(test_results)):
    print(i)
    c_t = [[0,0],[0,0]]
    for col in test_results.columns.to_list():
        #print("*************colname",col)
        if not (np.isnan(test_results.loc[i-1,col])) and not (np.isnan(test_results.loc[i,col])):
            #print(int(test_results.loc[i-1,col]),int(test_results.loc[i,col]))
            if (abs(int(test_results.loc[i-1,col]))>1 ) or (abs(int(test_results.loc[i,col]))>1 ):
                print("out of range error")
                print('col name', col)
                print(int(test_results.loc[i - 1, col]), int(test_results.loc[i, col]))
            c[int(test_results.loc[i-1,col])][int(test_results.loc[i,col])] += 1
            c_t[int(test_results.loc[i - 1, col])][int(test_results.loc[i, col])] += 1
    c_ts.append(c_t)
print(c)
print(c_ts)

RR = [x[0][0] for x in c_ts]
RB = [x[0][1] for x in c_ts]
BR = [x[1][0] for x in c_ts]
BB = [x[1][1] for x in c_ts]
R = [r+b for r,b in zip(RR,BR)]
B = [r+b for r,b in zip(RB,BB)]

RR_pct = [x[0][0]/np.sum(np.array(x))*100 for x in c_ts]
RB_pct = [x[0][1]/np.sum(np.array(x))*100 for x in c_ts]
BR_pct = [x[1][0]/np.sum(np.array(x))*100 for x in c_ts]
BB_pct = [x[1][1]/np.sum(np.array(x))*100 for x in c_ts]
R_pct = [r+b for r,b in zip(RR_pct,BR_pct)]
B_pct = [r+b for r,b in zip(BB_pct,RB_pct)]

years = np.arange(start=2006, stop = 2023)
flip_df = pd.DataFrame(data = {"RR":RR,"RB":RB,"BR":BR,"BB":BB, \
                               "RR%":RR_pct,"RB%":RB_pct,"BR%":BR_pct,"BB%":BB_pct, \
                               "R":R,"B":B,"R%":R_pct,"B%":B_pct}, \
                       index=years)
print(flip_df.head(16))

flip_df.iloc[:-1,[4,5,6,7]].plot(kind='line',color = ["red","purple","pink","blue"])
plt.show()
flip_df.iloc[:-1,[10,11]].plot(kind='line',color=['red','blue'])
plt.show()
flip_df.to_csv('flip_df.csv')

b = 1
#
# Todo
# 1. check whether variable names change (only for last year! lol)
# 2. check if coding changes for party (nope)
# 3. need col for date (have col, convert to proper date? or make year col)(done)
# 3. contiguous list of precinct ids--need the precincts that exist today (to be useful)(2023 data screwed up; no precinctUid)
# 3.5 use nameparser to split name for 2023 (/Users/charleskramer/Documents/RachelLevy2023/post_election_2023/post_election_2023.py)
# 3.75 2023 precinct name format changes? (no "}")
# 3.9 need at least two contiguous non nan obs df[].isna().sum() > df length-2 (16)
# 4. demographic info by precinct; translate census block -> precinct (interpolate?)
#     is there a utility to calc percentage overlap of two geographic areas? shapely? (https://shapely.readthedocs.io/en/stable/manual.html)
# geopandas has geo joins