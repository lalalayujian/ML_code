"""
高斯混合模型，是描述数据分布的生成概率模型
可以对简单的数据聚类，更多是用于生成数据
"""


import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ========================================高斯混合聚类==================================================
def test_gmm_cluster():
    X, y = make_blobs(n_features=13, n_samples=1000, centers=4, cluster_std=[2,5,3,5], random_state=100)
    X = StandardScaler().fit_transform(X)
    # 多维数据，根据聚类指标选择超参数n_clusters
    for i, k in enumerate((2,3,4,5)):
        # cluster = KMeans(n_clusters=k, max_iter=500, random_state=100)
        cluster = GaussianMixture(n_components=k, random_state=100)
        y_pred = cluster.fit_predict(X)
        # 轮廓系数，介于(-1,1)，越接近1越好，
        score_si = metrics.silhouette_score(X, y_pred)
        # calinski_harabaz分数，越大越好
        # 一般来说，Silhouette Coefficient要比Calinski-Harabasz Index的结果准确一些，但轮廓系数计算复杂度更高
        score_ch = metrics.calinski_harabasz_score(X, y_pred)
        print(k, score_si, score_ch)

    # 降维后画图看聚类效果
    # cluster = KMeans(n_clusters=4, max_iter=500, random_state=100)
    cluster = GaussianMixture(n_components=4, random_state=100)
    y_pred = cluster.fit_predict(X)
    data = pd.DataFrame(X)
    data['pre_class'] = y_pred
    de = TSNE(n_components=2, random_state=0).fit_transform(X)
    # de = PCA(n_components=2).fit_transform(X)
    # d = de[data['pre_class']==0]
    # plt.plot(d[:, 0], d[:, 1], 'r.')
    # d = de[data['pre_class']==1]
    # plt.plot(d[:, 0], d[:, 1], 'go')
    # d = de[data['pre_class']==2]
    # plt.plot(d[:, 0], d[:, 1], 'b*')
    # d = de[data['pre_class']==3]
    # plt.plot(d[:, 0], d[:, 1], 'y+')
    plt.scatter(de[:, 0], de[:, 1], c=y_pred)
    plt.show()

# test_gmm_cluster()


# ==========================================高斯混合模型生成数据===================================================
# 从根本上说，GMM是描述数据分布的生成概率模型，可模拟数据的分布用于生成新样本
def test_gmm_generate():
    Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)

    # 1.聚类效果不好
    cluster = GaussianMixture(n_components=2, random_state=100)
    y_pred = cluster.fit_predict(Xmoon)
    plt.scatter(Xmoon[:, 0], Xmoon[:, 1], c=y_pred)
    plt.show()

    # 2.生成模型
    # 确定高斯混合的成分个数，根据AIC或BIC选择
    models = [GaussianMixture(n_components=i, random_state=100).fit(Xmoon) for i in range(2, 15)]
    ax = plt.subplot()
    ax.plot(range(2, 15), [gmm.bic(Xmoon) for gmm in models], label='BIC')
    ax.plot(range(2, 15), [gmm.aic(Xmoon) for gmm in models], label='AIC')
    plt.legend()
    plt.show()

    # 根据创建的高斯混合分布，采样生成新数据
    model = GaussianMixture(n_components=10, random_state=100)
    model.fit(Xmoon, ymoon)
    print('converged: {}'.format(model.converged_))  # 模型是否收敛
    Xnew = model.sample(n_samples=400)[0]  # 返回新样本的数据和属于哪个高斯成分
    plt.scatter(Xnew[:, 0], Xnew[:, 1])
    plt.show()

# test_gmm_generate()


def gmm_digits():
    """
    利用gmm模型生成手写数字
    """
    digits = load_digits()
    print(digits.data.shape)  # (1797, 64)
    def plot_digits(data):
        fig, ax = plt.subplots(5, 5, figsize=(8, 8),
                               subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i, axi in enumerate(ax.flat):
            im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
            im.set_clim(0, 16)
        plt.show()
    plot_digits(digits.data)

    # 降维，保存99%的方差信息
    pca = PCA(n_components=0.99, whiten=True)
    data = pca.fit_transform(digits.data)
    print('pca shape: {}'.format(data.shape))

    # 确定gmm模型
    # models = [GaussianMixture(n_components=i, random_state=0).fit(data) for i in range(40, 150, 10)]
    # ax = plt.subplot()
    # ax.plot(range(40, 150, 10), [gmm.bic(data) for gmm in models], label='BIC')
    # ax.plot(range(40, 150, 10), [gmm.aic(data) for gmm in models], label='AIC')
    # plt.legend()
    # plt.show()

    model = GaussianMixture(n_components=130, random_state=0)
    model.fit(data)
    print('converged: {}'.format(model.converged_))  # 模型是否收敛
    Xnew = model.sample(n_samples=25)[0]  # 返回新样本的数据和属于哪个高斯成分
    # pca的逆转换还原数据格式
    digits_new = pca.inverse_transform(Xnew)
    plot_digits(digits_new)
