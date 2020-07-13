from collections import Counter
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 生成比例为9：1的二分类数据集
x, y  = make_classification(n_samples=10000, n_features=100, n_informative=80, weights=[0.9,0.1], random_state=0)
print(y.shape, (y==1).sum())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
print('训练集样本分布：', Counter(y_train).items())

rfc = RandomForestClassifier(n_estimators=10, random_state=10)
rfc.fit(x_train, y_train)
auc = roc_auc_score(y_test, rfc.predict_proba(x_test)[:,1])
print('auc: ', auc)

# 权重法
rfc = RandomForestClassifier(n_estimators=10, class_weight={0:1, 1:10}, random_state=10)
rfc.fit(x_train, y_train)
auc = roc_auc_score(y_test, rfc.predict_proba(x_test)[:,1])
print('auc: ', auc)

# =============================欠采样====================================
# from imblearn.under_sampling import RandomUnderSampler
# us = RandomUnderSampler(random_state=2019)
# x, y = us.fit_sample(x_train, y_train)
# print('通过欠采样后')
# print('采样后训练集样本分布：', Counter(y).items())
# rfc = RandomForestClassifier(n_estimators=10, random_state=10)
# rfc.fit(x, y)
# auc = roc_auc_score(y_test, rfc.predict_proba(x_test)[:,1])
# print('欠采样后auc: ', auc)

# =============================过采样====================================
# from imblearn.over_sampling import RandomOverSampler
# os = RandomOverSampler(random_state=2019)
# x, y = os.fit_sample(x_train, y_train)
# print('通过过采样后')
# print('采样后训练集样本分布：', Counter(y).items())
# rfc = RandomForestClassifier(n_estimators=10, random_state=10)
# rfc.fit(x, y)
# auc = roc_auc_score(y_test, rfc.predict_proba(x_test)[:,1])
# print('过采样后auc: ', auc)

# =============================smote采样=================================
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2019)
x, y = sm.fit_sample(x_train, y_train)
print('通过SMOTE采样后')
print('采样后训练集样本分布：', Counter(y).items())
rfc = RandomForestClassifier(n_estimators=10, random_state=10)
rfc.fit(x, y)
auc = roc_auc_score(y_test, rfc.predict_proba(x_test)[:,1])
print('SMOTE auc: ', auc)

