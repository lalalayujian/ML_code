import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100)
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error

x, y = make_regression(n_features=25, n_samples=5000, n_targets=1, noise=3, random_state=0)
print(y)


# 定义评估指标
def my_score(model, x, y):
    score = np.sqrt(-cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5))
    return score

# 单模型
lr = LinearRegression()
lasso = Lasso()
ridge = Ridge()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
clfs = [lr, lasso, ridge]  # , rf, gbr, 'RF', 'GBR'
names = ['LR', 'Lasso', 'Ridge']
for name, model in zip(names, clfs):
    score = my_score(model, x, y)
    print('{}: {:.6f} {:.4f}'.format(name, score.mean(), score.std()))


"""blending"""
"""一部分作为训练基模型，另一部分来生成新特征作为次级模型的输入。若不切割，直接用训练集训练的模型，将训练集转换，会过拟合。"""
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin


class Blending(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        new_x = np.zeros((x_test.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            """依次训练各个单模型，最终形成3个特征"""
            model.fit(x_train, y_train)
            new_x[:, i] = model.predict(x_test)
        self.meta_model.fit(new_x, y_test)
        return self

    def predict(self, X):
        new_x = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            new_x[:, i] = model.predict(X)
        pred = self.meta_model.predict(new_x)
        return pred

    def validation(self, x_train, y_train, x_test, y_test):
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_train, y_train, test_size=0.5)
        new_train = np.zeros((new_x_test.shape[0], len(self.base_models)))
        new_test = np.zeros((x_test.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            """依次训练各个单模型，最终形成3个特征"""
            model.fit(new_x_train, new_y_train)
            new_train[:, i] = model.predict(new_x_test)
            """对于测试集，直接用这k个模型的预测值作为新的特征"""
            new_test[:, i] = model.predict(x_test)
        self.meta_model.fit(new_train, new_y_test)
        pred = self.meta_model.predict(new_test)
        score = np.sqrt(mean_squared_error(y_test, pred))
        return score


blending = Blending(clfs, gbr)
cv_score = my_score(blending, x, y)
print(cv_score.mean())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
score = blending.validation(x_train, y_train, x_test, y_test)
print(score)