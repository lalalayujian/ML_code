"""
高斯朴素贝叶斯：假设特征的类条件分布服从高斯分布
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

data = load_digits()
X = data.data
y = data.target
print(X.shape)
print(np.unique(y))
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

gnb = GaussianNB()
gnb.fit(Xtrain, ytrain)
score = gnb.score(Xtest, ytest)
print('test accuracy: {}'.format(score))
# print('test cm: {}'.format(confusion_matrix(ytest, gnb.predict(Xtest))))

# 概率校准
cgn = CalibratedClassifierCV(gnb, cv=5)
cgn.fit(Xtrain, ytrain)
score = cgn.score(Xtest, ytest)
print('Calibrated test accuracy: {}'.format(score))
