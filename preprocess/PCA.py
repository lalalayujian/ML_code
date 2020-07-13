"""
对降维的效果评估，通常是比较降维前后模型的性能
若降到二维或者三维，可以直观看降维效果
"""


# ======================降维实现高维数据的可视化=========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# iris = load_iris()
# y = iris.target
# x = iris.data
#
# pca = PCA(n_components=2)
# x_dr = pca.fit_transform(x)
# # 降维后每个新特征向量上的可解释性方差的大小
# print(pca.explained_variance_)
# # 降维后每个新特征向量上的可解释方差贡献率
# print(pca.explained_variance_ratio_)
# print(pca.components_)  # 新特征向量矩阵v

# colors=['red', 'black', 'orange']
# plt.figure()
# for i in [0, 1, 2]:
#     plt.scatter(x_dr[y==i, 0], x_dr[y==i, 1], alpha=.7,
#                 c=colors[i], label=iris.target_names[i])
# plt.legend()
# plt.title('PCA of IRIS dataset')
# plt.show()
# plt.figure()
# plt.scatter(x_dr[:, 0], x_dr[:, 1], c=y)
# plt.show()

# ==================选取超参数n_components==================
# 方法一：累积可解释方差贡献率曲线
# n_components为None默认返回min(X.shape)个特征，一般样本量都会大于特征数目
# 此时相当于转换了新特征空间，但未减少特征数目
# pca_line = PCA().fit(x)
# plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
# plt.xticks([1,2,3,4])#这是为了限制坐标轴显示为整数
# plt.xlabel("numberofcomponentsafterdimensionreduction")
# plt.ylabel("cumulativeexplainedvarianceratio")
# plt.show()

# 方法二：极大似然估计自选超参数
# pca_mle=PCA(n_components="mle")
# pca_mle=pca_mle.fit_transform(x)
# print(pca_mle.explained_variance_ratio_)

# 方法三：按信息量占比选超参数
# pca_f = PCA(n_components=0.97, svd_solver="full")
# X_f = pca_f.fit_transform(x)
# print(pca_f.explained_variance_ratio_)

# =============================人脸数据看降维效果==========================
# from sklearn.datasets import fetch_lfw_people
#
# faces = fetch_lfw_people(min_faces_per_person=5)
# print(faces.data.shape, faces.images.shape)
# x = faces.data
#
# fig, axes = plt.subplots(4, 5, figsize=(8, 4), subplot_kw={'xticks':[], 'yticks':[]})
# # plt.show()
# # 展示原始图片
# for i, ax in enumerate(axes.flat):
#     ax.imshow(faces.images[i, :, :], cmap='gray')
# plt.show()
#
# pca = PCA(150).fit(x)
# v = pca.components_
#
# # 展示降维后主成分特征空间
# fig, axes = plt.subplots(3,5, figsize=(8,4), subplot_kw={'xticks':[], 'yticks':[]})
# for i, ax in enumerate(axes.flat):
#     ax.imshow(v[i, :].reshape(62,47), cmap='gray')
# plt.show()

# ==============================PCA做噪音过滤=================================
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

def plot_digits(data):
    fig, axes = plt.subplots(4,10,figsize=(10,4), subplot_kw = {"xticks":[],"yticks":[]})
    for i, ax in enumerate(axes.flat):
         ax.imshow(data[i].reshape(8,8),cmap="binary")
    plt.show()
plot_digits(digits.data)

# ======================降维实现对数据降噪=========================
# 加噪音
np.random.RandomState(42)
noisy = np.random.normal(digits.data, 2)
plot_digits(noisy)

pca = PCA(0.5).fit(noisy)
X_dr = pca.transform(noisy)
print(X_dr.shape)

# inverse_transform逆转降维结果，实现降噪
# 降维时部分信息已被舍弃，在逆转的时候，即便维度升高，原数据中被舍弃的信息也不可能再回来了。所以，降维不是完全可逆的。
without_noise = pca.inverse_transform(X_dr)
plot_digits(without_noise)

# =========================================降维+分类=================================
# 在手写数据集上(42000,784)，KNN跑一遍30min，RF15s，且KNN效果比RF好（但维度太高距离计算太慢）
# PCA + KNN的效果高于特征选择+RF，也高于PCA + RF
