#!/usr/bin/python3 -tt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
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
pca_max = int(n*0.8)

df_train = pd.read_csv("Training-sp.csv")
df_train = df_train.pivot(index='rowid',columns='term',values='tfidf')
df_train = df_train.fillna(value=0)

pca = PCA(n_components=pca_max)
df_pca = pca.fit_transform(df_train)
#print(pca.explained_variance_ratio_)

result = {
    'components' : [],
    'SVM_score' : [],
    'SVM_accuracy' : [],
    'RF_score' : [],
    'RF_accuracy' : [],
    'PLR_score' : [],
    'PLR_accuracy' : [],
}

for p in range(3,n+1):
    result['components'].append(p)

    X_train = df_pca[:,0:p]

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
print(result_df)


### the line chart of the accuracy and scores on the validation sets
plt.figure(figsize=(400/96,400/96),dpi=96)
plt.plot(result_df.index,result_df['SVM_score'],'b-',linewidth=1.0,label='SVM score')
plt.plot(result_df.index,result_df['SVM_accuracy'],'b:',label='SVM accuracy')
plt.plot(result_df.index,result_df['RF_score'],'r-',label='RF score')
plt.plot(result_df.index,result_df['RF_accuracy'],'r:',label='RF accuracy')
plt.plot(result_df.index,result_df['PLR_score'],'g-',label='PLR score')
plt.plot(result_df.index,result_df['PLR_accuracy'],'g:',label='PLR accuracy')
plt.legend()
plt.xlabel('number of principal components')
plt.tight_layout()  # when producing a small image 
plt.savefig('reduction-pca.png',dpi=96)
