#!/usr/bin/python3 -tt

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def score(estimator,X,y):
    scores = cross_val_score(estimator, X, y, cv=10, scoring='accuracy')
    return scores.mean()

def accuracy(estimator,X,y):
    estimator.fit(X,y)
    return metrics.accuracy_score(estimator.predict(X),y)

''' ---------- main ---------- '''
evaluation = pd.read_csv("evaluation.csv")
y_train = evaluation['EVAL'].astype('category')
n = evaluation.shape[0]

df_train = pd.read_csv("Training-sp.csv")
df_train = df_train.pivot(index='rowid',columns='term',values='tfidf')
df_train = df_train.fillna(value=0)

non_nzv = pd.Series(df_train.columns)
non_nzv = non_nzv[list(df_train.apply(lambda x: n-Counter(x)[0],axis=0) > 1)]

high_variance = df_train[non_nzv].var().sort_values(ascending=False).index
df_train = df_train[high_variance]

result = {
    'components' : [],
    'SVM_score' : [],
    'SVM_accuracy' : [],
    'RF_score' : [],
    'RF_accuracy' : [],
    'PLR_score' : [],
    'PLR_accuracy' : [],
}

for p in range(3,2*n+1):
    result['components'].append(p)

    X_train = df_train.ix[:,0:p]

    estimator_svc = SVC(coef0=0, kernel="sigmoid", gamma=0.1, C=2.56)
    result['SVM_score'].append(score(estimator_svc,X_train,y_train))
    result['SVM_accuracy'].append(accuracy(estimator_svc,X_train,y_train))

    estimator_rf = RandomForestClassifier(n_estimators=15,max_features='auto',random_state=1)
    result['RF_score'].append(score(estimator_rf,X_train,y_train))
    result['RF_accuracy'].append(accuracy(estimator_rf,X_train,y_train))

    estimator_plr = LogisticRegression(penalty='l2',class_weight={1:4,-1:6}, C=0.16)
    result['PLR_score'].append(score(estimator_plr,X_train,y_train))
    result['PLR_accuracy'].append(accuracy(estimator_plr,X_train,y_train))


result_df = pd.DataFrame(result,index=result['components'],columns=['SVM_score','SVM_accuracy','RF_score','RF_accuracy','PLR_score','PLR_accuracy'])
print("Result")
print(result_df)

plt.figure(figsize=(400/96,400/96),dpi=96)
plt.plot(result_df.index,result_df['SVM_score'],'b-',linewidth=1.0,label='SVM score')
plt.plot(result_df.index,result_df['SVM_accuracy'],'b:',label='SVM accuracy')
plt.plot(result_df.index,result_df['RF_score'],'r-',label='RF score')
plt.plot(result_df.index,result_df['RF_accuracy'],'r:',label='RF accuracy')
plt.plot(result_df.index,result_df['PLR_score'],'g-',label='PLR score')
plt.plot(result_df.index,result_df['PLR_accuracy'],'g:',label='PLR accuracy')
plt.legend(loc='upper left')
plt.xlabel('number of predictors')
plt.tight_layout()
plt.savefig('reduction-variance-2.png',dpi=96)
