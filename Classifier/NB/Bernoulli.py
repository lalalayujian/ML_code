"""
伯努利朴素贝叶斯：假设特征的类条件分布服从伯努利分布（0-1分布）
常用于文本分类，特征为单词出现与否向量
"""

import numpy as np
from sklearn.datasets import make_blobs, load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
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
bnb = BernoulliNB()
bnb.fit(X_train_, y_train)
print('class count : {}'.format(bnb.class_count_))
print('class prior : {}'.format(np.exp(bnb.class_log_prior_)))
print('feature_log_prob_:\n {}'.format(bnb.feature_log_prob_)) # 伯努利分布的参数（每个特征出现的概率）[n_classes, n_features]
# 样本x要判别的概率：分子为p(c)*p(feature=1|c)^x*(1-p(feature=1|c))^(1-x)
print('test accuracy: {}'.format(bnb.score(X_test_, y_test)))
print('test brier_score_loss: {}'.format(brier_score_loss(y_test,
                                                          bnb.predict_proba(X_test_)[:, 1],
                                                          pos_label=1)))

# 分类值建模，使用阈值粗暴二值化
print('分类值建模')
bnb = BernoulliNB(binarize=0.5)
bnb.fit(X_train_, y_train)
print('test accuracy: {}'.format(bnb.score(X_test_, y_test)))
print('test brier_score_loss: {}'.format(brier_score_loss(y_test,
                                                          bnb.predict_proba(X_test_)[:, 1],
                                                          pos_label=1)))
