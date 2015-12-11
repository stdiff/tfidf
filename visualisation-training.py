#!/usr/bin/python3 -tt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import operator ### for an assignment in a lambda
from sklearn.manifold import TSNE

def show_histogram(evaluation,df_train):
    '''
    Produce an image of the histogram of tfidf
    '''

    ### id_to_ev: rowid -> EVAL
    id_to_ev = {}
    evaluation.apply(
        lambda row: operator.setitem(id_to_ev,row['rowid'],row['EVAL']),
        axis=1)

    ### converts df['rowid'] into corresponding labels
    ev_list = []
    for i in range(df_train.shape[0]):
        ev_list.append(id_to_ev[df_train.ix[i,'rowid']])
    ev_list = pd.Series(ev_list)

    ### histogram of tfidf
    plt.figure(figsize=(400/96,400/96),dpi=96)
    plt.hist(
        [df_train['tfidf'][ev_list == k] for k in [-1,1]],
        stacked=True,
        label=['-1','+1'],
        color=['r','b'])
    plt.legend()
    plt.title("Histogram of tfidf")
    plt.tight_layout()
    plt.savefig('histogram-of-tfidf.png',dpi=96)

def plot_data(evaluation,df_train):
    '''
    2-dimensional visualisation via t-SNE
    '''

    sne = TSNE(n_components=2, random_state=2)
    np.set_printoptions(suppress=True)
    sne_xy = sne.fit_transform(df_train)

    ev_color = ['r','g','g']
    ev_label = ['-1','0','+1']

    plt.figure(figsize=(400/96,400/96),dpi=96)
    for i in range(evaluation.shape[0]): 
        plt.scatter(sne_xy[i,0], # x-coord   
                    sne_xy[i,1], # y-coord
                    alpha=0.5,
                    color=ev_color[evaluation.ix[i,'EVAL']+1],
                    label=ev_label[evaluation.ix[i,'EVAL']+1],
        )
        plt.annotate(i,
                     xy=(sne_xy[i,0],sne_xy[i,1]),
                     xytext=(4,-4), # offset
                     alpha=0.5,
                     textcoords='offset points',
                     color=ev_color[evaluation.ix[i,'EVAL']+1],
        )
    plt.title('Projected Labelled Items')
    plt.tight_layout()
    plt.savefig('labelled-documents.png',dpi=96)


''' ---------- main ---------- '''
evaluation = pd.read_csv("evaluation.csv")

df_train = pd.read_csv("Training-sp.csv")
show_histogram(evaluation,df_train)

df_train = df_train.pivot(index='rowid',columns='term',values='tfidf')
df_train = df_train.fillna(value=0)
plot_data(evaluation,df_train)
