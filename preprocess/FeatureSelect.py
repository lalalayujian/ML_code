import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=10000, n_features=300, n_redundant=50, n_informative=100, n_classes=2, random_state=10)
rfc = RandomForestClassifier(n_estimators=10, random_state=20)
score = cross_val_score(rfc, x, y, cv=5).mean()
print(score)

# ======================过滤法=======================
# 方差。使用阈值为0或很小的方差过滤，来优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法
# 一般数据维度高时，才用方差进行粗略过滤（因为粗略所以可不标准化），然后使用其他方法
# threshold = np.percentile(x.var(axis=0), 30)
# x_fsvar = VarianceThreshold(threshold=threshold).fit_transform(x)
# print(x_fsvar.shape)
# score =cross_val_score(rfc, x_fsvar, y, cv=5).mean()
# print(score)

# 卡方，计算非负特征和标签之间的卡方统计量
# 可通过显著性水平确定显著相关的特征数
# chivalue, p_chi = chi2(x_fsvar, y)
# k = chivalue.shape[0] - (p_chi > 0.05).sum()
# x_fschi = SelectKBest(chi2, k=k).fit_transform(x_fsvar, y)
# score =cross_val_score(rfc, x_fschi, y, cv=5).mean()
# print(score)

# F检验，捕捉每个特征与标签之间的线性关系
# f, p_f = f_classif(x_fsvar, y)
# k = f.shape[0] - (p_f > 0.05).sum()
# x_fsf = SelectKBest(f_classif, k=k).fit_transform(x_fsvar, y)
# score =cross_val_score(rfc, x_fsf, y, cv=5).mean()
# print(score)

# 互信息，捕捉每个特征与标签之间的任意关系
# 特征与目标的互信息量的估计，这个估计量在[0,1]取值，0表示两个变量独立，1表示两个变量完全相关
# result=mutual_info_classif(x_fsvar,y)
# k = result.shape[0] - (result <= 0).sum()
# x_fsmi = SelectKBest(mutual_info_classif, k=k).fit_transform(x_fsvar, y)
# score =cross_val_score(rfc, x_fsmi, y, cv=5).mean()
# print(k, score)

# ======================包装法=======================
# RFE递归消除，多次训练计算成本高，但取很少特征可以达到很好的效果
# select_n = range(1, 301, 50)
# scores = []
# for i in select_n:
#     x_rfe = RFE(rfc, n_features_to_select=i, step=50).fit_transform(x, y)
#     score = cross_val_score(rfc, x_rfe, y, cv=5).mean()
#     scores.append(score)
# plt.plot(select_n, scores)
# plt.xticks(select_n)
# plt.show()

# select_rfe = RFE(rfc, n_features_to_select=51, step=50)
# x_fsrfe = select_rfe.fit_transform(x, y)
# score =cross_val_score(rfc, x_fsrfe, y, cv=5).mean()
# print(x_fsrfe.shape[1], score)

# =======================嵌入法=======================
# 采用树模型，具有特征重要性，按照阈值删除不重要的特征
# thresholds = np.linspace(0, rfc.fit(x, y).feature_importances_.max(), 10)
# thresholds = np.linspace(0.003, 0.005, 10)
# scores = []
# for i in thresholds:
#     x_embedded = SelectFromModel(rfc, threshold=i).fit_transform(x, y)
#     score = cross_val_score(rfc, x_embedded, y, cv=5).mean()
#     scores.append(score)
# plt.plot(thresholds, scores)
# plt.xticks(thresholds)
# plt.show()
# threshold=0.00344
select_model = SelectFromModel(rfc, threshold=None)
x_embedded = select_model.fit_transform(x, y)
print(select_model.threshold_)  # 阈值None默认均值
score =cross_val_score(rfc, x_embedded, y, cv=5).mean()
print(x_embedded.shape[1], score)

# 采样正则化模型，选取系数大的特征
# select_model = SelectFromModel(LogisticRegression(C=1, random_state=0))
# x_embedded = select_model.fit_transform(x, y)
# print(select_model.threshold_)  # 阈值None默认系数绝对值的均值，正则化为l1时默认1e-5
# lrc = LogisticRegression(C=1, random_state=0)
# score =cross_val_score(lrc, x_embedded, y, cv=5).mean()
# print(x_embedded.shape[1], score)
# 可调整惩罚项系数，选取特征
# C = np.arange(0.01,10.01,0.5)
# for i in C:
#     LR_ = LogisticRegression(solver="liblinear",C=i,random_state=420)
#     X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(x, y)
#     score = cross_val_score(LR_, X_embedded, y, cv=10).mean()
#     print(x_embedded.shape[1], score)
