import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sklearn functions and models
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


def run_models(substance, data_cut, data_cut_int):
    substancies = ['alcohol','anphet', 'amyl', 'benzo', 'caffeine', 'cannabis',
              'chocolate', 'cocaine', 'crack', 'ecstasy', 'heroine', 'ketamine',
              'legal_h', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa']
    #substancies.remove(substance)
    #data_cut.drop(substancies, inplace = True)
    #data_cut_int.drop(substancies, inplace = True)

    print(substance.upper() + '\n')

    #logistic regression
    X_train, X_test, y_train, y_test = train_test_split(data_cut.drop(substancies, axis = 1),
                                                    data_cut[substance], test_size=0.3, random_state=0)

    model = LogisticRegression(random_state = 0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)
    print(f"Logistic regression accuracy score: {acc1}")

    #spec1 = specificity(y_test, y_pred) #not sure if it works!
    #print(f"Logistic regression specificity score: {spec1}")

    #cm1 = confusion_matrix(y_test, y_pred)

    #K-Fold
    cv = KFold(n_splits=6, random_state=1, shuffle=True) # more than 6 --> overfit
    scores = cross_val_score(model, data_cut.drop(substancies, axis = 1), data_cut[substance], scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Logistic regression k- fold accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    for i in ['linear', 'poly', 'rbf', 'sigmoid']: #try all different Kernel method
        print(f'SVM method: {i} \n')
        model = SVC(kernel = i, random_state = 0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc2 = accuracy_score(y_test, y_pred)
        print(f"Accuracy of kernel {i} is: {acc2}")
        #spec1 = specificity(y_test, y_pred)
        #print(f"Specificity score: {spec1}") #not sure it is correct
        cm1 = confusion_matrix(y_test, y_pred)
        cv = KFold(n_splits=5, random_state=1, shuffle=True) # more than 6 --> overfit
        scores = cross_val_score(model, data_cut.drop(substancies, axis = 1), data_cut[substance], scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('SVM accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        print('\n')

    X_train, X_test, y_train, y_test = train_test_split(data_cut_int.drop(substancies, axis = 1),
                                                    data_cut_int[substance], test_size=0.3, random_state=0)

    model = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print(y_train)
    acc3 = accuracy_score(y_test, y_pred)
    print(f"Neighboors classifier accuracy score: {acc3}")
    #spec1 = specificity(y_test, y_pred)
    #print(f"Neighboors classifier specificity score: {spec1}")

    cv = KFold(n_splits=5, random_state=1, shuffle=True) # more than 6 --> overfit
    scores = cross_val_score(model, data_cut_int.drop(substancies, axis = 1), data_cut_int[substance], scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Neighboors classifier k-fold accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc4 = accuracy_score(y_test, y_pred)
    print(f"Decision tree accuracy score: {acc4}")

    #spec1 = specificity(y_test, y_pred)
    #print(f"Decision tree specificity score: {spec1}")

    cv = KFold(n_splits=5, random_state=1, shuffle=True) # more than 6 --> overfit
    scores = cross_val_score(model, data_cut_int.drop(substancies, axis = 1), data_cut_int['cocaine'], scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Decision tree k-fold accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # XG-Boost
    print('XG_Boost \n')
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    X = data_cut.drop(substancies, axis = 1)[:1200]
    y = data_cut[substance][:1200]

    xgb_model.fit(X,y)
    X_test = data_cut.drop(substancies, axis = 1)[1200:]

    y_pred = xgb_model.predict(X_test)
    y_test = data_cut[substance][1200:].tolist()

    print(f'N train : {len(X)}; N test : {len(X_test)}')
    a = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            a += 1
    acc_xg = a/len(y_pred)
    print(f"XG-Boost accuracy score: {acc_xg}")

if __name__ == '__main__':
    data_cut = pd.read_csv('data_processed/data_cut.csv')
    data_cut.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)

    data_cut_int = pd.read_csv('data_processed/data_cut_int.csv')

    data_cut_int.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)


    run_models('cannabis', data_cut, data_cut_int)
