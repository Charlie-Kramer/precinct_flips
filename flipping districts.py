#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# In[71]:


df = pd.read_csv("voting_acs_df.csv").dropna()
df2 = df.loc[:,["2020Flip","PctMale","PctWhite","MedAge","PctForn","PctPoverty","PctBroadband","PctMedicaid"]]
df2.hist(column="2020Flip")


# In[21]:


y = np.array(df2.loc[:,"2020Flip"])
X = np.array(df2.loc[:,["PctMale","PctWhite","MedAge","PctForn","PctPoverty","PctBroadband","PctMedicaid"]])
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)


# In[74]:


n = len(y)
print(len(y))
test_frac = .8
n_trials = 100
errs = []

nums = [500,1750,250]
fracs = [x/sum(nums) for x in nums]
print(nums,sum(nums),fracs)

for _ in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)

    knn.fit(X_train,y_train)
    y_fit = knn.predict(X_test)
    err = y_fit != y_test
    print(sum(err)/n)
    errs.append(sum(err)/n)

