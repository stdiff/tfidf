#!/usr/bin/python3 -tt
from collections import defaultdict # for non-existing key of dict
import operator 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


### compute the preference vector
pref_vec = df[label==1].sum() - df[label==-1].sum() 


### compute the scalar products of the preference vector to labelled documents
score = pd.Series(np.dot(df,pref_vec),name='score',index=df.index)


### histogram of scores on the training set
plt.figure(figsize=(400/96,400/96),dpi=96)
plt.hist([score[label==1],score[label==-1]],stacked=True,label=['+1','-1'])
plt.legend()
plt.title("Histogram of scalar products on the training set")
plt.tight_layout()
plt.savefig('histo-scalar-1.png',dpi=96)


### histogram of scores on the test set
plt.figure(figsize=(400/96,400/96),dpi=96)
plt.hist([score[label==0]])
plt.title("Histogram of scalar products on the test set")
plt.tight_layout()
plt.savefig('histo-scalar-2.png',dpi=96)


### save the evaluation for the test set
score_test = score[label==0]
score_test.name = 'SP'
score_test = normalise(score_test)
result = pd.read_csv("evaluation-test.csv",index_col=0)
result = pd.concat([result,score_test],axis=1)
result.to_csv("evaluation-test.csv")

