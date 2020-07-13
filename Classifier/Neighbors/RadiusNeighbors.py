"""
限定半径最近邻：只在固定半径范围内查找最近邻
"""


import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import matplotlib.pyplot as plt


X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.show()

clf = RadiusNeighborsClassifier(radius=1, weights='distance', outlier_label=1)
# 若样本在指定半径范围内没有近邻样本，指定outlier_label。如果不指定，遇到异常点会报错
clf.fit(X, y)
print('score: {}'.format(clf.score(X, y)))

# 查看目标样本的近邻样本（距离+位置）
# print(clf.radius_neighbors(X[0,:].reshape(1, -1), return_distance=True))
# 查看目标样本的近邻图（稀疏矩阵，位置+距离或者连通）
# print(clf.radius_neighbors_graph(X[0].reshape(1, -1), mode='distance'))


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
