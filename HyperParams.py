"""
调参的三大方法：网格搜索、随机搜索、贝叶斯优化
"""


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import clone
import hyperopt
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 生成分类数据集（不平衡）
data = make_classification(n_samples=6000, n_features=10, n_classes=2, weights=[0.65, 0.35])
train_x, test_x, train_y, test_y = train_test_split(data[0], data[1], test_size=0.3)


print('缺省模型'.center(40, '-'))
start = time.time()
model = GradientBoostingClassifier()
model.fit(train_x, train_y)
train_time = time.time() - start
print('training time is {:.4f} seconds'.format(train_time))
pre_y = model.predict(test_x)
auc = roc_auc_score(test_y, pre_y)
print('test auc: {}'.format(auc))

'''
print('网格搜索'.center(40, '-'))
param_grid = {
    'n_estimators': [30, 40, 50],
    'learning_rate': [0.05,0.08,0.1],
    'subsample': [0.7,0.8,0.9],
    'max_features': range(5, 10, 2),
    'max_depth': range(3, 6),
    'max_leaf_nodes': range(60,80,5)
}
start = time.time()
model = GradientBoostingClassifier()
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(train_x, train_y)
train_time = time.time() - start
print('training time is {:.4f} seconds'.format(train_time))
print(clf.best_params_)
best_model = clone(clf.best_estimator_)
best_model.fit(train_x, train_y)
pre_y = best_model.predict(test_x)
auc = roc_auc_score(test_y, pre_y)
print('test auc: {}'.format(auc))
'''


print('贝叶斯优化'.center(40, '-'))
from hyperopt import hp, tpe, fmin, partial, pyll
# 目标函数
def objective(params):
    model = GradientBoostingClassifier(
        loss=params['loss'],
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_features=int(params['max_features']) * 2 + 5,
        max_leaf_nodes=int(params['max_leaf_nodes']),
        max_depth=params['max_depth'] + 3
    )
    cv_score = cross_val_score(model, train_x, train_y, cv=5, scoring='roc_auc')
    loss = 1 - cv_score.mean()
    return loss

# 域空间
space = {
    'loss':hp.choice('loss', ['deviance', 'exponential']),  # 类别变量,选项可以是list或者tuple.options可以是嵌套的表达式，用于组成条件参数
    'n_estimators': hp.quniform('n_estimators', 20, 50, 2),  # 离散均匀分布（在整数空间上均匀分布）
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),  # 连续对数均匀分布（在浮点数空间中的对数尺度上均匀分布）
    'subsample': hp.uniform('subsample', 0.5, 1),  # 连续均匀分布（在浮点数空间上均匀分布）
    'max_features': hp.randint('max_features', 3),  # [0,1,2]
    'max_depth': hp.randint('max_depth', 5),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes',20, 80, 5),
}
# 从参数空间内采样
# from hyperopt.pyll.stochastic import sample
# print(sample(space))

# 优化算法(Hyperopt 目前只支持TPE树形评估器和随机搜索)
# algo = tpe.suggest  # rand.suggest
algo = partial(tpe.suggest, n_startup_jobs=10)
# 结果历史数据
# bayes_trials = hyperopt.Trials()

# 优化
max_evals = 2  # 迭代优化轮数
start = time.time()
best = fmin(fn=objective, space=space, algo=algo, max_evals=max_evals)
train_time = time.time() - start
print(best)
print('training time is {:.4f} seconds'.format(train_time))

# 测试集表现
def best_test(best):
    best_model = GradientBoostingClassifier(
        loss=['deviance', 'exponential'][best['loss']],
        n_estimators=int(best['n_estimators']),
        learning_rate=best['learning_rate'],
        subsample=best['subsample'],
        max_features=int(best['max_features']) * 2 + 5,
        max_leaf_nodes=int(best['max_leaf_nodes']),
        max_depth=best['max_depth'] + 3
    )
    best_model.fit(train_x, train_y)
    pre_y = best_model.predict(test_x)
    auc = roc_auc_score(test_y, pre_y)
    print('test auc: {}'.format(auc))
best_test(best)


print('随机搜索'.center(40, '-'))
#sklearn有接口
param_grid = {
    'n_estimators': range(20, 50, 2),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=500)),
    'subsample': list(np.linspace(0.5, 1, 50)),
    'max_features': range(5, 10, 2),
    'max_depth': range(3, 7),
    'max_leaf_nodes': range(20,80,5)
}

# plt.hist(param_grid['learning_rate'])
# # plt.xlabel('Learning Rate')
# # plt.ylabel('Count')
# # plt.title('Learning Rate Distribution')
# # plt.show()

def random_model(params):
    model = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_features=params['max_features'],
        max_depth=params['max_depth'],
        max_leaf_nodes=params['max_leaf_nodes']
    )
    return model

result = pd.DataFrame(columns=['loss', 'params', 'iteration'])
max_evals = 200  # 迭代优化轮数
start = time.time()
for i in range(max_evals):
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    model = random_model(params)
    cv_score = cross_val_score(model, train_x, train_y, cv=5, scoring='roc_auc')
    loss = 1 - cv_score.mean()
    result.loc[i,:] = [loss, params, i]
train_time = time.time() - start
print('training time is {:.4f} seconds'.format(train_time))
result.sort_values('loss', ascending=True, inplace=True)
result.reset_index(inplace=True, drop=True)
print(result.head())

# 测试集表现
best_params = result.loc[0, 'params']
model = random_model(best_params)
model.fit(train_x, train_y)
pre_y = model.predict(test_x)
auc = roc_auc_score(test_y, pre_y)
print('test auc: {}'.format(auc))

