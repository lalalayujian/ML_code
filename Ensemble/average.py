import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split

"""解析地址"""
# import cpca
# data = pd.read_csv('C:/Users/bdfus_003/Desktop/address.csv', encoding='gbk')
# df = cpca.transform(data['FAMILYADD'], cut=False)
# print(df)

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

# Average集成
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, models, weight):
        self.models = models
        self.weight = weight

    def fit(self, X, y):
        self.models = [clone(m) for m in self.models]
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        pred = np.array([model.predict(X) for model in self.models])  # 3个模型3行
        w_pred = np.multiply(pred.T, np.array(self.weight))  # 广播机制(5000*3)*(3,) --> (5000,3)
        w = w_pred.sum(axis=1)
        return w

weight_avg = AverageWeight(clfs, [0.2,0.2,0.6])
score = my_score(weight_avg, x, y)
print(score.mean())