"""
标签扩散算法：基于图的半监督学习算法，越相似的样本越容易传播
样本集是由少量已标注的数据和大量未标注的数据组成
"""


import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import time


X, y = load_digits(return_X_y=True)
print('data shape: {}'.format(X.shape))

# 留部分测试数据
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# 生成有未标记样本的数据集
rng = np.random.RandomState(0)
random_unlabeled = rng.rand(len(y_train)) < 0.8
# 未标记样本的标签设置为-1
y_train[random_unlabeled] = -1

# 调参gamma
# for i in [0.005, 0.01, 0.1, 0.5, 1]:
#     model = LabelPropagation(kernel='rbf', gamma=i)
#     model.fit(x_train, y_train)
#     print(i, accuracy_score(y_test, model.predict(x_test)))

model = LabelSpreading(kernel='rbf', gamma=0.01)
# model = LabelPropagation(kernel='rbf', gamma=0.01)
model.fit(x_train, y_train)

print('===========y===============')
print(y_test)
print('===========y_pred===============')
y_pred = model.predict(x_test)
print(y_pred)
print('=======confusion_matrix=======')
print(confusion_matrix(y_test, y_pred))
print('accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(model.label_distributions_)