"""
层次聚类，试图在不同层次对数据集进行划分，从而形成树型的聚类结构
节约内存、聚类速度快、可以发现异常点
对高维特征的数据集聚类效果不好。如果数据的分布簇不是类似于超球体，或者不凸，效果也不好
"""


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.3, 0.2, 0.4, 0.3], random_state=9)
cluster = Birch()
y_pred = cluster.fit_predict(X)
print('original calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

# 调参聚为几类
for i in range(2,6):
    cluster = Birch(n_clusters=i)
    y_pred = cluster.fit_predict(X)
    print('{}, calinski_harabasz_score: {}'.format(i, calinski_harabasz_score(X, y_pred)))

# 调整CF树的参数，进一步提升聚类效果
max_score = 0
best_param = None
params = {'th': [0.1, 0.2, 0.3, 0.4, 0.5],
          'br': [20, 25, 30, 40, 50]}
for th in params['th']:
    for br in params['br']:
        cluster = Birch(n_clusters=4, threshold=th, branching_factor=br)
        y_pred = cluster.fit_predict(X)
        score = calinski_harabasz_score(X, y_pred)
        print(th, br, score)
        if score >= max_score:
            max_score = score
            best_param = {'th': th, 'br': br}
print(best_param, max_score)

# 最终的聚类结果
cluster = Birch(n_clusters=4, threshold=0.4, branching_factor=50)
y_pred = cluster.fit_predict(X)
print('best calinski_harabasz_score: {}'.format(calinski_harabasz_score(X, y_pred)))
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

