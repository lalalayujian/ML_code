"""
K近邻
搜索近邻时，样本和特征较少时可以蛮力搜索，多时应考虑KD树或者球树（更适合特征多时）
"""


import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.show()

clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
# 默认p=2的闵可夫斯基距离（欧式距离），metric=cosine为余弦距离, metric='haversine'为经纬度球体距离
clf.fit(X, y)
print('score: {}'.format(clf.score(X, y)))

# 查看目标样本的近邻样本（距离+位置）
# print(clf.kneighbors(X[0,:].reshape(1, -1), return_distance=True))
# 查看目标样本的近邻图（稀疏矩阵，位置+距离或者连通）
# print(clf.kneighbors_graph(X[0].reshape(1, -1), mode='distance'))


# 可视化预测的效果（决策边界）
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 确认训练集的边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 生成随机数据来做测试集，然后作预测
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
new_x = np.c_[xx.ravel(), yy.ravel()]
y_pred = clf.predict(new_x)

# 画出测试集数据
ax = plt.subplot()
ax.pcolormesh(xx, yy, y_pred.reshape(xx.shape), cmap=cmap_light)
# 也画出所有的训练集数据
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'distance')" )
plt.show()
