"""
标签传播算法：基于图的半监督学习算法，越相似的样本越容易传播
样本集是由少量已标注的数据和大量未标注的数据组成
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import time


def iris_semi():
    X, y = load_iris(return_X_y=True)
    print('data shape: {}'.format(X.shape))

    # 降维，方便可视化
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    # 设置画布
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    fig = plt.figure()
    for i, threshold in enumerate([0.3, 0.5, 0.8, 1]):
        new_y = y.copy()
        if threshold < 1:
            rng = np.random.RandomState(0)
            random_unlabeled = rng.rand(len(y)) <= threshold  # 0-1的随机数，小于等于threshold返回True
            # 未标记样本的标签设置为-1
            new_y[random_unlabeled] = -1

            model_name = 'LabelPropagation'
            model = LabelPropagation(kernel='rbf', gamma=20)
        else:
            model_name = 'SVC'
            model = SVC()
        model.fit(X, new_y)

        # 生成网格数据点
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        new_x = np.c_[xx.ravel(), yy.ravel()]
        z = model.predict(new_x)

        # 画出网格数据的预测值
        ax = fig.add_subplot(2, 2, i + 1)
        ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cmap_light, alpha=0.5)
        # 画出真实数据分布
        ax.scatter(X[:, 0], X[:,1], c=y, cmap=cmap_bold)
        ax.set_title('{}, {}% data'.format(model_name, threshold*100))

    plt.show()


iris_semi()