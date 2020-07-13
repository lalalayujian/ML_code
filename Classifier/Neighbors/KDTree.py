"""
KD树数据结构，常用于空间数据划分，查找最近邻
"""


import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


rng = np.random.RandomState(0)
X = rng.random_sample((100, 4))
kdtree = KDTree(X, leaf_size=5)

# 查询目标样本的k近邻样本的距离和索引
dist, ind = kdtree.query(X[:1], k=3)
print(ind)  # k近邻样本的索引
print(dist)  # k近邻样本的距离

# 查询目标样本的指定半径内的近邻
ind = kdtree.query_radius(X[:1], r=0.3)
print(ind)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
from matplotlib.patches import Circle
ax.add_patch(Circle([X[:1, 0], X[:1, 1]], r=0.3, color='r', fill=False))
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X[:1, 0], X[:1, 1], c='r')
plt.show()