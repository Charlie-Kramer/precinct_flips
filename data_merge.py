#
# merge data from (1) precinct voting outcomes, (2) precinct-to-tract mapper, (3) tract ACS data
#
import pickle
import pandas as pd

df = pd.read_csv('df0.csv')
print(df.head())
cols = [c[:4] for c in df.columns.tolist()]
coldict = dict(zip(df.columns.tolist(),cols))
df.rename(columns=coldict,inplace=True)
c0 = df.columns.tolist()[0]
df.rename(columns={c0:'PrecinctUid'},inplace=True)


with open('prec2tract.txt', 'rb') as file:
  prec2tract = pickle.loads(file.read())
#note: this dictionary only includes 2019-2020 precincts

df["2020Flip"] = df["2020"]-df["2019"]

voting_df = df.loc[:,["PrecinctUid","2020Flip","2020","2019"]].dropna()

prec = voting_df["PrecinctUid"].tolist()
tract = [int(prec2tract.get(p,-99999)) for p in prec]
voting_df['Tract'] = tract

acs_df = pd.read_csv('acs_data.csv')
voting_acs_df = voting_df.merge(acs_df,how='left',left_on="Tract",right_on="GEOID")


test_data = voting_acs_df.iloc[0,:]
print(test_data)
test_data = voting_acs_df.iloc[372,:]
print(test_data)
# both check out
#
# add 4-category variable: DD, DR, RD, RR
#
def four_cat(x2019, x2020):
  if   ((x2019==0) and (x2020==0)):
    return "RR"
  elif ((x2019==0) and (x2020==1)):
    return "RD"
  elif ((x2019==1) and (x2020==0)):
    return "DR"
  elif ((x2019==1) and (x2020==1)):
    return "DD"

voting_acs_df["flip4cat"] = voting_acs_df.apply(lambda x: four_cat(x["2019"],x["2020"]), axis=1)

voting_acs_df["Pop"] = voting_acs_df["B01001_001E"]
voting_acs_df["PctMale"] = voting_acs_df["B01001_002E"]/voting_acs_df["Pop"]
voting_acs_df["PctWhite"] = voting_acs_df['B01001A_001E']/voting_acs_df['Pop']
voting_acs_df["MedAge"] = voting_acs_df['B01002_001E']
med_age = voting_acs_df["MedAge"].loc[voting_acs_df["MedAge"]>0].median()
voting_acs_df["MedAge"] = voting_acs_df["MedAge"].replace(to_replace=-666666666.00000,value=med_age)
voting_acs_df["PctForn"] = voting_acs_df['B05002_013E']/voting_acs_df["Pop"]
voting_acs_df["PctPoverty"] = voting_acs_df['B06012_002E']/voting_acs_df["Pop"]
voting_acs_df["PctBroadband"] = voting_acs_df['B28002_004E']/voting_acs_df['Pop']
voting_acs_df["PctMedicaid"] = voting_acs_df["C27007_001E"]/voting_acs_df['Pop']

voting_acs_df.to_csv("voting_acs_df.csv")