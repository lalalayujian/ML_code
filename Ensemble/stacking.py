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
gbr =GradientBoostingRegressor()
clfs = [lr, lasso, ridge]  #, rf, gbr, 'RF', 'GBR'
names = ['LR', 'Lasso', 'Ridge']
for name, model in zip(names, clfs):
    score = my_score(model, x, y)
    print('{}: {:.6f} {:.4f}'.format(name, score.mean(), score.std()))


"""stacking"""
"""Base Model 之间的相关性要尽可能的小，Base Model 之间的性能表现不能差距太大"""
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
class Stacking(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        # self.base_models_ = [list() for x in self.base_models]  此处应为正确的，有时间改正
		# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
        new_x = np.zeros((X.shape[0],len(self.base_models)))
        for i, model in enumerate(self.base_models):
            """依次训练各个基模型，最终形成(基模型数量)个特征"""
            for train_index, test_index in self.kf.split(X, y):
                """使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征"""
                # instance = clone(model)
                # self.base_models_[i].append(instance)
                x_train, y_train = X[train_index], y[train_index]
                x_test, y_test = X[test_index], y[test_index]
                # model = clone(model)
                model.fit(x_train, y_train)
                new_x[test_index, i] = model.predict(x_test)
        self.meta_model.fit(new_x, y)
        return self

    def predict(self, X):
        new_x = np.zeros((X.shape[0],len(self.base_models)))
        for i, model in enumerate(self.base_models):
            new_x[:, i] = model.predict(X)
        pred = self.meta_model.predict(new_x)
        return pred

    def validation(self, x_train, y_train, x_test, y_test):
        new_train = np.zeros((x_train.shape[0], len(self.base_models)))
        new_test = np.zeros((x_test.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            """依次训练各个基模型，最终形成(基模型数量)个特征"""
            test_agg = np.zeros((x_test.shape[0], 5))
            for j, (train_index, test_index) in enumerate(self.kf.split(x_train, y_train)):
                """使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征"""
                model.fit(x_train[train_index], y_train[train_index])
                new_train[test_index, i] = model.predict(x_train[test_index])
                test_agg[:, j] = model.predict(x_test)
            """对于测试集，直接用这几个基模型的预测值均值作为新的特征"""
            new_test[:, i] = test_agg.mean(axis=1)
        self.meta_model.fit(new_train, y_train)
        pred = self.meta_model.predict(new_test)
        score = np.sqrt(mean_squared_error(y_test, pred))
        return score


stacking = Stacking(clfs, gbr)
cv_score = my_score(stacking, x, y)
print(cv_score.mean())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
score = stacking.validation(x_train, y_train, x_test, y_test)
print(score)