#!/usr/bin/python3 -tt

from collections import Counter
import numpy as np
import pandas as pd
import re

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def predict_df(df_train,y_train,df_test,suffix):
    columns = ['SVM','RF','PLR']
    if suffix:
        columns = [ "%s_%s" % (x,suffix) for x in columns ]

    result = pd.DataFrame()

    estimator_svc = SVC(coef0=0, kernel="sigmoid", gamma=0.1, C=2.56)
    estimator_svc.fit(df_train,y_train)
    result[columns[0]] = estimator_svc.predict(df_test)

    estimator_rf = RandomForestClassifier(n_estimators=15,max_features='auto',random_state=1)
    estimator_rf.fit(df_train,y_train)
    result[columns[1]] = estimator_rf.predict(df_test)
    
    estimator_plr = LogisticRegression(penalty='l2',class_weight={1:4,-1:6}, C=0.16)
    estimator_plr.fit(df_train,y_train)
    result[columns[2]] = estimator_plr.predict_proba(df_test)[:,1]

    return result


''' ---------- main ---------- '''
evaluation = pd.read_csv("evaluation.csv")
y_train = evaluation['EVAL'].astype('category')
n = evaluation.shape[0]

df = pd.read_csv("Training-sp.csv")
df['rowid'] = [ "ev%s" % rowid for rowid in df['rowid']]
df = pd.concat([df,pd.read_csv("Test-sp.csv")]) 
df = df.pivot(index='rowid',columns='term',values='tfidf')
df = df.fillna(value=0)

### split the data frame into training data and test data
in_train = [ bool(re.search('^ev',str(x))) for x in list(df.index)]
in_test = [ not x for x in in_train ]
df_train = df[in_train]
df_test = df[in_test]

result = pd.DataFrame()
result = pd.concat([result,predict_df(df_train,y_train,df_test,'')],axis=1)


### PCA
pca = PCA(n_components=10)
df_pca = pd.DataFrame(pca.fit_transform(df),index=df.index)
df_pca_train = df_pca[in_train]
df_pca_test = df_pca[in_test]

result = pd.concat([result,predict_df(df_pca_train.ix[:,0:5],y_train,df_pca_test.ix[:,0:5],'PCA5')],axis=1)
result = pd.concat([result,predict_df(df_pca_train.ix[:,0:10],y_train,df_pca_test.ix[:,0:10],'PCA10')],axis=1)


### predictors with high variance
non_nzv = pd.Series(df.columns)
non_nzv = non_nzv[list(df_train.apply(lambda x: n-Counter(x)[0],axis=0) > 1)]

high_variance = df_train.var().sort_values(ascending=False).index
hign_variance = [x in non_nzv for x in high_variance]

df_hv_train = df_train[high_variance].ix[:,0:70]
df_hv_test = df_test[high_variance].ix[:,0:70]

result = pd.concat([result,predict_df(df_hv_train,y_train,df_hv_test,'hv')],axis=1)

### save the prediction to a csv file
result.index = df_test.index
result.to_csv("evaluation-test.csv")

