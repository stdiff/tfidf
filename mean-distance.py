#!/usr/bin/python3 -tt
from collections import defaultdict # for non-existing key of dict
import operator 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dist(df):
    '''
    compute the matrix of distances 
    This function does not care any symmetry. 
    '''
    n = df.shape[0]
    items = df.index
    X = np.zeros([n,n],float)
    for i in range(n):
        for j in range(n):
            diff = df.ix[i,:] - df.ix[j,:]
            X[i,j] = np.sqrt(np.dot(diff,diff))
    return pd.DataFrame(X,index=items,columns=items)

def normalise(s):
    ''' 
    Normalise a sequence (series) into [0,1]
    '''
    M = s.max()
    m = s.min()
    return (s-m)/(M-m)

### id_to_ev: rowid -> EVAL
id_to_ev = defaultdict(int)
evaluation = pd.read_csv("evaluation.csv")
evaluation.apply(
    lambda row: operator.setitem(id_to_ev,"ev%s" % row['rowid'] ,row['EVAL']),
    axis=1)


### data frame whose rows are vectors corresponding to the labelled documents
df = pd.read_csv("Training-sp.csv")
df['rowid'] = [ "ev%s" % rowid for rowid in df['rowid']]
df = pd.concat([df,pd.read_csv("Test-sp.csv")]) 
df = df.pivot(index='rowid',columns='term',values='tfidf')
df = df.fillna(value=0)

### series of labels
label = df.apply(lambda row: id_to_ev[row.name], axis=1)

### distance matrix (data frame)
dist_matrix = dist(df)



n = df.shape[0]
label_pos = np.zeros(n)
label_neg = np.zeros(n)
for i in range(n):
    if label[i] == 1:
        label_pos[i] = 1
    elif label[i] == -1:
        label_neg[i] = 1

label_pos = (label_pos/label_pos.sum())
label_neg = (label_neg/label_neg.sum())

### compute the mean distances 
mean_dist_neg = np.dot(dist_matrix,label_neg) # numerator
mean_dist_pos = np.dot(dist_matrix,label_pos) # denominator

### the score (ratio of mean distances)
mean_dist_score = pd.Series(mean_dist_neg/mean_dist_pos, index=df.index)
mean_dist_score = mean_dist_score[ label == 0 ]

plt.figure(figsize=(400/96,400/96),dpi=96)
plt.hist([mean_dist_score])
plt.title('Histogram of the ratio of mean distances')
plt.tight_layout()
plt.savefig('histo-mean-distance.png',dpi=96)


### save the evaluation for the test set
mean_dist_score = normalise(mean_dist_score)
mean_dist_score.name = 'MD'
result = pd.read_csv("evaluation_test.csv",index_col=0)
result = pd.concat([result,mean_dist_score],axis=1)
result.to_csv("evaluation_test.csv")
