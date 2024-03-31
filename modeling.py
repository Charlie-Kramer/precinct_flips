import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             matthews_corrcoef, balanced_accuracy_score, make_scorer)
import time

balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

def score_it(y_test, y_pred, model, label_list, plots=False):
    '''
    given y_test and y_pred, generate scoring metrics; model is label for model type, plots = produce confusion matrix plots
    '''
    print("scoring ",model)
    print("class numbers--test set")
    vc = pd.DataFrame(y_test).value_counts().sort_index()
    print(vc)
    print("F1 score", f1_score(y_test, y_pred))
    print("Matthews", matthews_corrcoef(y_test,y_pred))
    print("Balanced Accuracy", balanced_accuracy_score(y_test,y_pred))

    if plots:
        print("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8,5))
        cm = confusion_matrix(y_test, y_pred)

        cmp = ConfusionMatrixDisplay(cm,
                                 display_labels=label_list)
        cmp.plot(ax=ax)
        plt.title(model+", Unnormalized")
        plt.show()
        plt.savefig(model+"Unnorm"+label_list[0]+label_list[1]+".png")
    # normalize by 'true'

        fig, ax = plt.subplots(figsize=(8,5))
        cm = confusion_matrix(y_test, y_pred,normalize='true')
        cmp = ConfusionMatrixDisplay(cm,
                                 display_labels=label_list)
        cmp.plot(ax=ax)
        plt.title(model+", Normalized by True")
        plt.show()
        plt.savefig(model + "True" + label_list[0] + label_list[1]+".png")

        fig, ax = plt.subplots(figsize=(8,5))
        cm = confusion_matrix(y_test, y_pred, normalize='pred')
        cmp = ConfusionMatrixDisplay(cm,
                                 display_labels=label_list)
        cmp.plot(ax=ax)
        plt.title(model+", Normalized by Predicted")
        plt.show()
        plt.savefig(model + "Pred" + label_list[0] + label_list[1]+".png")

    return None

def fit_models(label_list, df):
    y = np.array(df.loc[:, "2020"])
    X = np.array(df.loc[:, ["PctMale", "PctWhite", "MedAge", "PctForn", "PctPoverty", "PctBroadband", "PctMedicaid"]])

    print("category value counts", pd.DataFrame(y).value_counts())

    test_size = 0.2
    X_train_unscaled, X_test_unscaled, y_train0, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_)
    print("category value counts-test set", pd.DataFrame(y_test).value_counts())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)

    k_ = 5
    smote = SMOTE(k_neighbors = k_)
    X_train,y_train = smote.fit_resample(X_train_scaled,y_train0)

    n_cv = 10

    #
    # fit base model (KNN)--use CV
    #

    print("***********************")
    print("knn")
    print("***********************")
    model = "KNN"
    k_list = np.arange(1,20,1)
    k_grid = {'n_neighbors': k_list, "p" : [1,2], 'weights' : ["uniform", "distance"]}
    knn = KNeighborsClassifier(weights = 'uniform')
    knn_cv = GridSearchCV(knn, k_grid, cv = n_cv, scoring=balanced_accuracy_scorer)
    knn_cv.fit(X_train, y_train)
    print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
    print("Best score is {}".format(knn_cv.best_score_))
    #
    # random forest
    #
    print("***********************")
    print("random forest")
    print("***********************")
    model = "Random Forest"
    k_list = np.arange(20,200,10)
    k_grid = {"n_estimators" : k_list, "criterion" : ['gini', 'entropy', 'log_loss'],
              "min_samples_split" : [2,4,6], "min_samples_leaf": [1,3,5]}
    rf = RandomForestClassifier(random_state=seed_)
    rf_cv = GridSearchCV(rf, k_grid, cv = n_cv, scoring=balanced_accuracy_scorer)
    rf_cv.fit(X_train, y_train)

    print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
    print("Best score is {}".format(rf_cv.best_score_))
    #
    # ADABoost
    #
    print("***********************")
    print("adaboost")
    print("***********************")
    model = "ADABoost"
    ab = AdaBoostClassifier(algorithm= "SAMME",random_state=seed_)
    k_grid = {"n_estimators": np.arange(5,100,5), 'learning_rate' : [.25, .75, 1, 2, 4]}
    ab_cv = GridSearchCV(ab, k_grid, cv = n_cv, scoring=balanced_accuracy_scorer)
    ab_cv.fit(X_train, y_train)
    print("Tuned AB Parameters: {}".format(ab_cv.best_params_))
    print("Best score is {}".format(ab_cv.best_score_))

    print("***********************")
    print("SVM")
    print("***********************")
    model = "SVM"

    svm = SVC(random_state=seed_,class_weight='balanced')
    k_grid = {"C":[.5,1,2],"kernel":['linear','poly','rbf','sigmoid'],
              "gamma":["scale","auto"]}
    svm_cv = GridSearchCV(svm,k_grid, cv = n_cv, scoring=balanced_accuracy_scorer)
    svm_cv.fit(X_train, y_train)
    print("Tuned SVM Parameters: {}".format(svm_cv.best_params_))
    print("Best score is {}".format(svm_cv.best_score_))

    print("***********************")
    print("NN")
    print("***********************")
    model = "nn"

    nn = MLPClassifier(random_state=seed_,max_iter=1000)
    k_grid = {"activation":["tanh","relu"],"hidden_layer_sizes":[50,100,200],
              "learning_rate":["constant","adaptive"]}
    nn_cv = GridSearchCV(nn,k_grid,cv=n_cv,scoring=balanced_accuracy_scorer)
    nn_cv.fit(X_train,y_train)
    print("Tuned NN Parameters: {}".format(nn_cv.best_params_))
    print("Best score is {}".format(nn_cv.best_score_))

    best_scores = {"knn": knn_cv.best_score_, "rf": rf_cv.best_score_,
                   "ab": ab_cv.best_score_, "svm": svm_cv.best_score_,
                   "nn": nn_cv.best_score_}
    best_params = {"knn": knn_cv.best_params_, "rf": rf_cv.best_params_,
                   "ab": ab_cv.best_params_, "svm": svm_cv.best_params_,
                   "nn": nn_cv.best_params_}

    return best_scores, best_params

def re_fit(best_params, df, model, label_list):
    y = np.array(df.loc[:, "2020"])
    X = np.array(df.loc[:, ["PctMale", "PctWhite", "MedAge", "PctForn", "PctPoverty", "PctBroadband", "PctMedicaid"]])

    print("category value counts", pd.DataFrame(y).value_counts())

    test_size = 0.2
    X_train_unscaled, X_test_unscaled, y_train0, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_)
    print("category value counts-test set", pd.DataFrame(y_test).value_counts())


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)

    k_ = 5
    smote = SMOTE(k_neighbors = k_)
    X_train,y_train = smote.fit_resample(X_train_scaled,y_train0)

    print("****")
    print(" re - fitting")
    if model == "knn":
        knn = KNeighborsClassifier(weights=best_params[model]['weights'],
                                   p = best_params[model]['p'],
                                   n_neighbors=best_params[model]["n_neighbors"])
        y_pred = knn.fit(X_train,y_train).predict(X_test)
        score_it(y_test, y_pred, model, label_list,plots=True)

    if model == "rf":
        rf = RandomForestClassifier(criterion=best_params[model]['criterion'],
                                    n_estimators=best_params[model]['n_estimators'],
                                    min_samples_split=best_params[model]['min_samples_split'],
                                    min_samples_leaf=best_params[model]['min_samples_leaf'],
                                    random_state=seed_)
        y_pred = rf.fit(X_train, y_train).predict(X_test)
        score_it(y_test, y_pred, model,label_list,plots=True)

    if model == "ab":
        ab = AdaBoostClassifier(algorithm= "SAMME", n_estimators = best_params[model]['n_estimators'],
                                learning_rate = best_params[model]['learning_rate'],
                                random_state=seed_)
        y_pred = ab.fit(X_train, y_train).predict(X_test)
        score_it(y_test, y_pred, model,label_list,plots=True)

    if model == "svm":
        svm = SVC()
        svm = SVC(random_state=seed_,class_weight='balanced',
                  C=best_params[model]["C"],kernel=best_params[model]['kernel'],
                  gamma=best_params[model]['gamma'])
        y_pred = svm.fit(X_train, y_train).predict(X_test)
        score_it(y_test, y_pred, model, label_list, plots=True)

    if model == "nn":
        nn = MLPClassifier(random_state=seed_,max_iter=1000,activation=best_params[model]["activation"],
            hidden_layer_sizes=best_params[model]["hidden_layer_sizes"],
            learning_rate=best_params[model]["learning_rate"])

        y_pred = nn.fit(X_train, y_train).predict(X_test)
        score_it(y_test, y_pred, model, label_list, plots=True)

start = time.time()
B = 30 # MC reps for smote
seed_ = 13

model_list = ["knn","rf","ab","svm","nn"]

#
# do first for base = D
#
print("***********************")
print("**********************************************")
print("base D")
print("**********************************************")
print("***********************")
df = pd.read_csv("~/Documents/precinct_flips/voting_acs_df.csv").dropna()

df = df.loc[df['2019']==1]
label_list = ["DR","DD"]

top_score = {"knn":0,"rf":0,"ab":0,"svm":0, "nn":0}
top_params = {}
scores = {"knn":[],"rf":[],"ab":[],"svm":[],"nn":[]}

for i in range(B):
    print("/////////////////////////////////////////////////////////////////////////////////////")
    print("smote iteration",i,'elapsed',time.time()-start)
    print("/////////////////////////////////////////////////////////////////////////////////////")
    best_score, best_params = fit_models(label_list, df)

    for model in model_list:
        scores[model].append(best_score[model])
        if (best_score[model] >= top_score[model]):
            top_score[model] = best_score[model]
            top_params[model] = best_params[model]

print("***********************")
print("**********************************************")
print("Best for base D")
print("***********************")
print("**********************************************")

for model in model_list:
    print("model "+model+" best score and params")
    print(best_score[model])
    print(best_params[model])
    re_fit(best_params,df,model,label_list)

dfn = pd.DataFrame.from_dict(scores)
dfn.to_csv("D_base_scores.csv")

#
# now for base = R
#
print("***********************")
print("**********************************************")
print("base R")
print("**********************************************")
print("***********************")

df = pd.read_csv("~/Documents/precinct_flips/voting_acs_df.csv").dropna()

df = df.loc[df['2019']==0]
label_list = ["RR","RD"]

top_score = {"knn":0,"rf":0,"ab":0,"svm":0,"nn":0}
top_params = {}
scores = {"knn":[],"rf":[],"ab":[],"svm":[],"nn":[]}

for i in range(B):
    print("smote iteration",i)
    best_score, best_params = fit_models(label_list, df)
    for model in model_list:
        scores[model].append(best_score[model])
        if (best_score[model] >= top_score[model]):
            top_score[model] = best_score[model]
            top_params[model] = best_params[model]

print("***********************")
print("**********************************************")
print("Best for base R")
print("***********************")
print("**********************************************")

for model in model_list:
    print(best_score[model])
    print(best_params[model])
    re_fit(best_params,df,model,label_list)

dfn = pd.DataFrame.from_dict(scores)
dfn.to_csv("R_base_scores.csv")

print("execution time = ",time.time()-start,"seconds")

