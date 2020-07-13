"""
多项式朴素贝叶斯：假设所有特征的类联合概率服从多项式分布，特征一般是次数
通常需要计数特征，或者出现与否这样的特征。
常用于文本分类，特征为单词计数向量，实际上小数（例如tf-idf）也可能有用
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss


X, y = make_blobs(n_samples=[500, 500],
                  centers=[[0.0, 0.0], [2.0, 2.0]],
                  cluster_std=[0.5, 0.5],
                  random_state=0, shuffle=False
                  )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 归一化，确保MultinomialNB输入数据无负数
mms = MinMaxScaler().fit(X_train)
X_train_ = mms.transform(X_train)
X_test_ = mms.transform(X_test)

print('连续值建模')
mnb = MultinomialNB()
mnb.fit(X_train_, y_train)
print('class count : {}'.format(mnb.class_count_))
print('class prior : {}'.format(np.exp(mnb.class_log_prior_)))
print('feature_log_prob_:\n {}'.format(mnb.feature_log_prob_)) # 多项式分布的参数（每次实验，特征出现的概率）[n_classes, n_features]
# 样本x要判别的概率：分子为p(c)*p(feature|c)^x
print('test accuracy: {}'.format(mnb.score(X_test_, y_test)))
print('test brier_score_loss: {}'.format(brier_score_loss(y_test,
                                                          mnb.predict_proba(X_test_)[:, 1],
                                                          pos_label=1)))
# 分箱，将数据转换为分类型
from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins=10, encode='onehot', strategy='quantile')  # 哑变量后也可以使用伯努利朴素贝叶斯
kbd.fit(X_train)
X_train_ = kbd.transform(X_train)
X_test_ = kbd.transform(X_test)

# 分类值建模，此处只为展示效果，并无实际意义
print('分类值建模')
mnb = MultinomialNB()
mnb.fit(X_train_, y_train)
print('test accuracy: {}'.format(mnb.score(X_test_, y_test)))
print('test brier_score_loss: {}'.format(brier_score_loss(y_test,
                                                          mnb.predict_proba(X_test_)[:, 1],
                                                          pos_label=1)))
