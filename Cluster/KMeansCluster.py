"""
KMeans聚类属于原型聚类
适用于凸数据集（两个样本点的线段上的样本也在数据集中）
对异常点敏感
"""


import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


'''
"""二维数据， 方便可视化，画聚类后结果图"""
X, y = make_blobs(n_features=2, n_samples=1000, centers=4, cluster_std=[2,1.5,1,1], random_state=100)

plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

X = StandardScaler().fit_transform(X)
cluster = KMeans(n_clusters=2, max_iter=1000, random_state=100)
cluster.fit(X)
y_pred = cluster.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

# 不同k值的聚类效果
for i, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,i+1)
    cluster = KMeans(n_clusters=k, max_iter=500, random_state=100)
    y_pred = cluster.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    score = metrics.calinski_harabasz_score(X, y_pred)
    plt.text(0.99, 0.01, ('k=%d, score=%.2f' %(k, score)),transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()

# 各类别下x1的密度曲线，可看出各类别的该变量的分布差异
data = pd.DataFrame(X, columns=['x1', 'x2'])
data['pre_class'] = y_pred
plt.figure()
data[data['pre_class']==0]['x1'].plot(kind='kde', label='0')
data[data['pre_class']==1]['x1'].plot(kind='kde', label='1')
data[data['pre_class']==2]['x1'].plot(kind='kde', label='2')
data[data['pre_class']==3]['x1'].plot(kind='kde', label='3')
plt.xlabel('x1')
plt.ylabel(u'密度')
plt.legend()
plt.show()
'''


"""多维数据，根据聚类指标选择超参数n_clusters"""
X, y = make_blobs(n_features=8, n_samples=1000, centers=4, cluster_std=[2,1.5,3,1], random_state=100)
X = StandardScaler().fit_transform(X)
for i, k in enumerate((2,3,4,5)):
    cluster = KMeans(n_clusters=k, max_iter=500, random_state=100)
    y_pred = cluster.fit_predict(X)
    # 轮廓系数，介于(-1,1)，越接近1越好，
    score_si = metrics.silhouette_score(X, y_pred)
    # calinski_harabaz分数，越大越好
    # 一般来说，Silhouette Coefficient要比Calinski-Harabasz Index的结果准确一些，但轮廓系数计算复杂度更高
    score_ch = metrics.calinski_harabasz_score(X, y_pred)
    print(k, score_si, score_ch)

# 降维后画图看聚类效果
cluster = KMeans(n_clusters=4, max_iter=500, random_state=100)
y_pred = cluster.fit_predict(X)
data = pd.DataFrame(X)
data['pre_class'] = y_pred
# de = TSNE(n_components=2).fit_transform(X)
de = PCA(n_components=2).fit_transform(X)
d = de[data['pre_class']==0]
plt.plot(d[:, 0], d[:, 1], 'r.')
d = de[data['pre_class']==1]
plt.plot(d[:, 0], d[:, 1], 'go')
d = de[data['pre_class']==2]
plt.plot(d[:, 0], d[:, 1], 'b*')
d = de[data['pre_class']==3]
plt.plot(d[:, 0], d[:, 1], 'y+')
plt.show()
