"""
分类朴素贝叶斯：假设所有特征的类条件概率服从离散分布
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss


X, y = make_blobs(n_samples=[500, 500],
                  centers=[[0.0, 0.0], [2.0, 2.0]],
                  cluster_std=[0.5, 0.5],
                  random_state=0, shuffle=False
                  )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 分箱，将数据转换为分类型
from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
kbd.fit(X_train)
X_train_ = kbd.transform(X_train)
X_test_ = kbd.transform(X_test)

print('分类值建模')
cnb = CategoricalNB()
cnb.fit(X_train_, y_train)
print('test accuracy: {}'.format(cnb.score(X_test_, y_test)))
print('test brier_score_loss: {}'.format(brier_score_loss(y_test,
                                                          cnb.predict_proba(X_test_)[:, 1],
                                                          pos_label=1)))
