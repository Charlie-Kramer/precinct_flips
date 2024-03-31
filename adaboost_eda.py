#
# choose best oversampling method based on adaboost f1 score
#
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN,SVMSMOTE
from sklearn.metrics import f1_score,confusion_matrix

df = pd.read_csv("voting_acs_df.csv").dropna()

y = np.array(df.loc[:,"2020Flip"])
X = np.array(df.loc[:,["PctMale","PctWhite","MedAge","PctForn","PctPoverty","PctBroadband","PctMedicaid"]])

seed_ = 2112

# split sample, do SMOTE on training sample only
# train_0 is pre-smote sample because in CV I will re-smote it
test_size = 0.2
X_train0, X_test, y_train0, y_test = train_test_split(X,y,test_size=test_size, random_state=seed_)

ab = AdaBoostClassifier(random_state=seed_,algorithm="SAMME")
#
# base case: no smoting
#
print("base")
y_pred = ab.fit(X_train0, y_train0).predict(X_test)
print(f1_score(y_test, y_pred, average='weighted'),len(X_train0),len(y_train0))

def scoring(smote_fun):
    scores = []
    for k_ in range(1,25):
        if smote_fun==ADASYN:
            smote = smote_fun(n_neighbors=k_)
        else:
            smote = smote_fun(k_neighbors=k_)
        X_train,y_train = smote.fit_resample(X_train0,y_train0)
        y_pred = ab.fit(X_train,y_train).predict(X_test)
        scores.append(f1_score(y_test,y_pred,average='weighted'))
    return np.max(scores),np.where(scores==np.max(scores)),np.mean(scores),np.std(scores)

B = 100

labels = ["SMOTE","BLSMOTE","SVMSMOTE","ADASYN"]
winners = []

for i in range(B):
    print("round ",i)
    smote_max, smote_k, _, _ = scoring(SMOTE)
    bl_smote_max, bl_smote_k, _, _ = scoring(BorderlineSMOTE)
    svm_smote_max, svm_smote_k, _, _ = scoring(SVMSMOTE)
    adasyn_smote_max, adasyn_smote_k, _, _ = scoring(ADASYN)

    maxes = sorted(zip([smote_max,bl_smote_max,svm_smote_max,adasyn_smote_max],labels,
                       [smote_k,bl_smote_k,svm_smote_k,adasyn_smote_k]),
                   reverse = True, key=lambda x: x[0])

    print(labels)
    print(maxes)
    print("winner ", maxes[0])
    winners.append(maxes[0])

df_smote = pd.DataFrame(winners)
df_smote.to_csv('df_smote_adaboost_cv.csv')

