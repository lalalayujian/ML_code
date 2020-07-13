import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier

"""生成数据集"""
# data, target = datasets.make_blobs(n_samples=20000, n_features=10, centers=2, random_state=0)
data, target = datasets.make_classification(n_samples=30000, n_features=10, n_classes=2, random_state=0, weights=[0.7,0.3])

"""切割数据集"""
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0, stratify=target)

""""bagging"""
clf = GaussianNB()
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
bg_clf = BaggingClassifier(GaussianNB(), n_estimators=20, max_features=5, max_samples=0.8, random_state=0)
bg_clf.fit(X_train, y_train)
print(roc_auc_score(y_test, bg_clf.predict_proba(X_test)[:, 1]))

"""voting"""
clf1 = LogisticRegression(random_state=0)
clf2 = GradientBoostingClassifier(random_state=0)
clf3 = RandomForestClassifier(random_state=0)

# hard是标签少数服从多数
ensemble_clf = VotingClassifier(estimators=[('LR', clf1), ('GB', clf2), ('RF', clf3)], voting='hard')
for label, clf in zip(['LR', 'GB', 'RF', 'ENSEMBLE'], [clf1, clf2, clf3, ensemble_clf]):
    print('%s val acc %.4f' % (label, cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()))

# soft是将概率加权平均，要求每个基模型都能预测概率
ensemble_clf = VotingClassifier(estimators=[('LR', clf1), ('GB', clf2), ('RF', clf3)], voting='soft')
for label, clf in zip(['LR', 'GB', 'RF', 'ENSEMBLE'], [clf1, clf2, clf3, ensemble_clf]):
    print('%s val auc %.4f' % (label, cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()))

# 设置加权的权重
ensemble_clf = VotingClassifier(estimators=[('LR', clf1), ('GB', clf2), ('RF', clf3)], voting='soft', weights=[2,1,2])
for label, clf in zip(['LR', 'GB', 'RF', 'ENSEMBLE'], [clf1, clf2, clf3, ensemble_clf]):
    print('%s val auc %.4f' % (label, cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc').mean()))

ensemble_clf.fit(X_train, y_train)
print(ensemble_clf.predict(X_test[0].reshape(1, -1)))
print(ensemble_clf.predict_proba(X_test[0].reshape(1, -1)))