"""
补集朴素贝叶斯：多项式朴素贝叶斯的改进，适合不平衡的数据集。
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score
from time import time
import datetime


X, y = make_blobs(n_samples=[50000, 500],
                  centers=[[0.0, 0.0], [5.0, 5.0]],
                  cluster_std=[3, 1],
                  random_state=0, shuffle=False
                  )

name = ["Multinomial","Gaussian","Bernoulli","Complement"]
models = [MultinomialNB(),GaussianNB(),BernoulliNB(),ComplementNB()]

for name, clf in zip(name,models):
    times = time()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
    #预处理    
    if name!= "Gaussian":
        kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
        Xtrain = kbs.transform(Xtrain)
        Xtest = kbs.transform(Xtest)
    clf.fit(Xtrain,Ytrain)
    y_pred = clf.predict(Xtest)
    proba = clf.predict_proba(Xtest)[:,1]
    score = clf.score(Xtest,Ytest)

    print(name)
    print("\tBrier:{:.3f}".format(brier_score_loss(Ytest,proba,pos_label=1)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\tRecall:{:.3f}".format(recall_score(Ytest,y_pred)))
    print("\tAUC:{:.3f}".format(roc_auc_score(Ytest,proba)))
    # print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))