#!/usr/bin/python3 -tt

import numpy as np
import pandas as pd
import json 

from sklearn.grid_search import GridSearchCV
from sklearn import metrics 

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def training(typ,val,model,df_train,y_train):
    print("model :", model)

    if model == 'svm':
        estimator = SVC()
        param_grid = {
            'C'      : [0.01*(2**d) for d in range(1,13)],
            'gamma'  : [0.01,0.03,0.1,0.3,1.0,1.3,10,13],
            'kernel' : ['rbf','linear','sigmoid','poly'],
            'coef0'  : [-1,-0.3,-0.1,-0.03,-0.01,0,0.01,0.03,0.1,0.3,1],
        }
    elif model == 'rf':
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators' : [10,15,20,25],
            'max_features' : [0.01*r for r in range(1,11)]
        }
    else: # plr
        estimator = LogisticRegression(solver='liblinear',penalty='l2')
        param_grid = {
            'C' : [0.01*(2**d) for d in range(1,13)],
            'class_weight' : [ {1:p, -1:10-p} for p in range(1,10)],
        }

    result = pd.DataFrame({'typ':[typ],'val':[val]},columns=['typ','val'])
    return(pd.concat([result,fit(model,estimator,param_grid,df_train,y_train)],axis=1))


def fit(model,estimator,param_grid,df_train,y_train):
    grid = GridSearchCV(estimator,param_grid,cv=10,scoring='accuracy')
    grid.fit(df_train,y_train)
    y_pred = pd.Series(grid.best_estimator_.predict(df_train),name='PRED')

    result = {
        'model'      : [model],
        'best_param' : [json.dumps(grid.best_params_)],
        'best_score' : [grid.best_score_],
        'accuracy'   : [metrics.accuracy_score(y_pred,y_train)]
    }
    return(pd.DataFrame(result,columns=['model','best_param','best_score','accuracy']))


''' ---------- main ---------- '''
evaluation = pd.read_csv("evaluation.csv")
y_train = evaluation['EVAL'].astype('category') # We assume the items in CSVs to be sorted by rowid.

result = pd.DataFrame(columns=['typ','val','model','best_param','best_score','accuracy'])
types = ['00','s0','0p','sp']
values = ['tf','idf','tfidf','ntfidf']

for typ in types:
    for val in values:
        print("type  :",typ)
        print("value :",val)
        df_train = pd.read_csv("Training-%s.csv" % typ)
        df_train = df_train.pivot(index='rowid',columns='term',values=val)
        df_train = df_train.fillna(value=0)

        result = pd.concat([result,training(typ,val,'svm',df_train,y_train)])
        result = pd.concat([result,training(typ,val,'rf',df_train,y_train)])
        result = pd.concat([result,training(typ,val,'plr',df_train,y_train)])

print(result.sort_values(by='best_score',ascending=False))
result.to_csv("classification-result.csv",index=False)

