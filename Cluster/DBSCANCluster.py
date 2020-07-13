"""
密度聚类，假设聚类结构可以用紧密程度来确定
可以对任意稠密数据集进行聚类，聚类的同时可以发现异常值。如果样本集的密度不均匀，聚类效果较差
参数eps，距离阈值过大，核心对象会变少，本来密度不可达的样本可能变得密度可达（不该是一类划为了一类），可能聚类类别数会变少
参数min_samples，对一定的eps，该参数过大，核心对象会变少，本来密度可达的样本可能不能密度可达，聚类类数会变多
"""


import numpy as np
from sklearn.cluster import DBSCAN, KMeans, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import calinski_harabasz_score, silhouette_score, homogeneity_score, davies_bouldin_score
import matplotlib.pyplot as plt


# Xmoon, ymoon = make_moons(n_samples=200, noise=.05, random_state=0)
# # 1.k-means聚类效果不好
# y_pred = KMeans(n_clusters=2, random_state=100).fit_predict(Xmoon)
# print('KMeans')
# print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(Xmoon, y_pred)))
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1], c=y_pred)
# plt.title('KMeans')
# plt.show()
#
#
# # 2.DBSCAN，需要调参
# # 默认参数聚出来一类，说明需要增加类别数，可以减小邻域距离阈值，使核心对象变多
# y_pred = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xmoon)
# print(np.unique(y_pred))
# print('DBSCAN')
# print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(Xmoon, y_pred)))
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1], c=y_pred)
# plt.title('DBSCAN')
# plt.show()


# ================环形数据============
X, y = make_circles(n_samples=500, random_state=0)
# 1.k-means聚类效果不好
y_pred = KMeans(n_clusters=2, random_state=100).fit_predict(X)
print('KMeans')
print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
print('\t homogeneity_score: {}'.format(silhouette_score(X, y_pred)))  # 同质性
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('KMeans')
plt.show()

# 2.DBSCAN，需要调参
# 默认参数聚出来一类，说明需要增加类别数，可以减小邻域距离阈值，使核心对象变多
y_pred = DBSCAN(eps=0.1, min_samples=5).fit_predict(X)
print('DBSCAN')
print(np.unique(y_pred))
print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
print('\t homogeneity_score: {}'.format(silhouette_score(X, y_pred)))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('DBSCAN')
plt.show()
# kmeans的calinski_harabasz_score比dbscan高，不符合事实。
# 原因：The Calinski-Harabasz index is generally higher for convex clusters than other concepts of clusters, such as density based clusters like those obtained through DBSCAN.
# 凸聚类：聚类结果中若每个簇都有一个凸包（用一个凸多边形围住所有的点），且凸包不相交，则称为凸聚类
# 数据不凸，DBSCAN的轮廓系数和Calinski-Harabasz效果不好
# Calinski-Harabasz分数的定义，很明显，它在“球状”聚类算法上效果最好，因为它奖励了聚类质心相距较远且类内接近的聚类，与DBSCAN不符


# for eps in [0.1, 0.2, 0.3, 0.4]:
#     for min_samples in [5,8]:
#         print(eps, min_samples)
#         cluster = DBSCAN(eps=eps, min_samples=min_samples)  # 默认参数聚出来一类，说明需要增加类别数，可以减小邻域距离阈值，使核心对象变多
#         y_pred = cluster.fit_predict(X)
#         print('DBSCAN')
#         print(np.unique(y_pred))
#         print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))



# ==========================自定义距离函数进行聚类============================
'''
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabaz_score
# KMeans 聚类
X = train_data.values[:, 2:4]
plt.scatter(train_data['经度'], train_data['纬度'])
plt.show()

def haversine(latlon1, latlon2):
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    # 转为弧度制
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat = lat2 -lat1
    dlon = lon2 - lon1

    h = np.sin(dlon/2)**2 + np.cos(lon2)*np.cos(lon1)*np.sin(dlat/2)**2
    r = 6371  # 地球半径
    distance = 2 * np.arcsin(np.sqrt(h)) * r
    return distance

from scipy.spatial.distance import pdist, squareform
distance_c = pdist(X, lambda u, v: haversine(u, v))  # 距离向量
distance_matrix = squareform(distance_c)  # 距离矩阵
clu = DBSCAN(eps=30, min_samples=10, metric='precomputed')
pred = clu.fit_predict(distance_matrix)
print(set(clu.labels_))
print(pred)

plt.scatter(X[:, 1], X[:, 0], c=pred)
plt.show()
score = calinski_harabaz_score(X, pred)
print(score)
train_data['cluster'] = pred

qd = pd.get_dummies(train_data['cluster'], prefix='cluster')
train_data = train_data.drop(['经度', '纬度'], axis=1)
train_data = pd.concat([train_data, qd], axis=1)
train_data = train_data.drop(['cluster'], axis=1)'''