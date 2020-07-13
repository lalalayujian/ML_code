"""
谱聚类
只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。
如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。
聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。
不需要事先标准化，Ncut拉普拉斯矩阵已标准化
"""


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering, KMeans, Birch, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score, homogeneity_score


X, y = make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
y_pred = SpectralClustering().fit_predict(X)
print('original calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
print('original silhouette_score: {}'.format(silhouette_score(X, y_pred)))

# for k in range(2,8):
#     # 使用高斯核，需要调gamma
#     for gamma in [0.01, 0.1, 1, 10]:
#         cluster = SpectralClustering(n_clusters=k, gamma=gamma)
#         y_pred = cluster.fit_predict(X)
#         print('n_clusters={}, gamma={}, calinski_harabasz_score: {}'.format(k, gamma, calinski_harabasz_score(X, y_pred)))
#         print('n_clusters={}, gamma={}, silhouette_score: {}'.format(k, gamma, silhouette_score(X, y_pred)))

cluster = SpectralClustering(n_clusters=5, gamma=0.1)
y_pred = cluster.fit_predict(X)
print('Spectral calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
print('Spectral silhouette_score: {}'.format(silhouette_score(X, y_pred)))

y_pred = DBSCAN(eps=2, min_samples=5).fit_predict(X)
print('DBSCAN')
print(np.unique(y_pred))
print('\t calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
print('\t homogeneity_score: {}'.format(silhouette_score(X, y_pred)))
