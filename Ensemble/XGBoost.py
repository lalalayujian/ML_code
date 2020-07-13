"""
    XGBoost：exterme gradient boosting
    希望每轮迭代得到最优的弱学习器，使用贪婪算法，分裂准则是损失最大程度减小，得到最优节点区域后，可直接计算叶子输出
    GBDT的优化，
"""


import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import KFold, train_test_split, learning_curve, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt


data = load_boston()
X = data.data
y = data.target
print(X.shape)
print('y mean: '.format(y.mean()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# ====================================第一种使用方式：sklearn接口=========================================
clf = xgb.XGBRegressor(n_estimators=180, max_depth=3, random_state=420)
clf.fit(X_train, y_train)

print('sklearn')
print('\t train r2: {}'.format(clf.score(X_train, y_train)))
print('\t train rmse: {}'.format(np.sqrt(mean_squared_error(y_train, clf.predict(X_train)))))
print('\t test r2: {}'.format(clf.score(X_test, y_test)))
print('\t test rmse: {}'.format(np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))))
# print('\t feature importance: {}'.format(clf.feature_importances_))

# 单个参数的学习曲线（可考虑泛化误差）
# rs = []
# # vs = []
# # ge = []  # Genelization error = bias**2 + var + 噪声
# axisx = range(50, 201, 10)
# cv = KFold(n_splits=5, shuffle = True, random_state=42)
# for i in axisx:
#     clf = xgb.XGBRegressor(n_estimators=i, random_state=420)
#     cv_result = cross_val_score(clf, X_train, y_train, cv=cv)
#     rs.append(cv_result.mean())
#     # vs.append(cv_result.var())
#     # ge.append((1 - cv_result.mean())**2 + cv_result.var())
# print(axisx[rs.index(max(rs))], max(rs))
# # print(axisx[vs.index(max(vs))], max(vs))
# # print(axisx[ge.index(max(ge))], max(ge))
# plt.figure(figsize=(20,5))
# plt.plot(axisx, rs, c='blue', label='XGB')
# #添加方差线
# # plt.plot(axisx,np.array(rs)+np.array(vs)*0.01,c="red",linestyle='-.')
# # plt.plot(axisx,np.array(rs)-np.array(vs)*0.01,c="red",linestyle='-.')
# plt.legend()
# plt.show()

# ====================================第二种使用方式：xgboost库===============================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, y_test)
params = {'eta': 0.1,
          'max_depth': 3,
          'silent': True,  # 是否隐藏建树过程'random_state': 420
          }
num_round = 180
xgbr = xgb.train(params, dtrain, num_boost_round=num_round)

print('xgboost')
print('\t train r2: {}'.format(r2_score(y_train, xgbr.predict(dtrain))))
print('\t train rmse: {}'.format(np.sqrt(mean_squared_error(y_train, xgbr.predict(dtrain)))))
print('\t test r2: {}'.format(r2_score(y_test, xgbr.predict(dtest))))
print('\t test rmse: {}'.format(np.sqrt(mean_squared_error(y_test, xgbr.predict(dtest)))))
# print('\t feature importance: {}'.format(clf.feature_importances_))

# 交叉验证，有每轮训练集和测试集结果
cv_result = xgb.cv(params, dtrain, num_round, nfold=5)
print(cv_result)
plt.figure(figsize=(20,5))
plt.grid()
plt.plot(range(1,181),cv_result.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),cv_result.iloc[:,2],c="orange",label="test,gamma=0")
plt.legend()
plt.show()

# 特征重要性
# print(xgbr.get_score())
# ax = xgb.plot_importance(xgbr, max_num_features=20)
# # fig = ax.figure
# # fig.set_size_inches(15, 20)
# plt.savefig('feature_XGB.jpg')
# plt.show()

# ================================================调参===============================================
# 网格搜索
# Step1. 学习率和估计器及其数目
# param_grid = {'n_estimator': range(50, 101, 10)}
# clf = xgb.XGBClassifier()
# gridcv = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')
# gridcv.fit(train_x, train_y)
# print(gridcv.best_params_)
# print(gridcv.best_score_)
# Step2. 树的最大深度、叶子的最小权重
# Step3. 控制分支的最小损失增益量gamma
# Step4. 子采样和特征采样
# Step1. 正则化

print('贝叶斯优化'.center(40, '-'))
from hyperopt import hp, tpe, fmin, partial, pyll, Trials

def objective(params):
    """目标函数，返回值应是类似于loss这样的越小越好的指标"""
    print(params)
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = params['max_depth'] + 3

    model = xgb.XGBClassifier(**params)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    loss = 1 - cv_score.mean()
    print('loss: {}'.format(loss))
    return loss

# 域空间
space = {
    'n_estimators': hp.quniform('n_estimators', 20, 50, 2),  # 离散均匀分布（在整数空间上均匀分布）
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),  # 连续对数均匀分布（在浮点数空间中的对数尺度上均匀分布）
    'subsample': hp.uniform('subsample', 0.6, 1),  # 连续均匀分布（在浮点数空间上均匀分布）
    'gamma': hp.randint('gamma', 3),  # [0,1,2]
    'max_depth': hp.randint('max_depth', 3),
    'min_child_weight': hp.randint('min_child_weight',5),
    'colsample_bytree':hp.uniform('colsample_bytree', 0.7, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}

# 优化算法(Hyperopt 目前只支持TPE树形评估器和随机搜索)
# algo = tpe.suggest  # rand.suggest
algo = partial(tpe.suggest, n_startup_jobs=10)

# 贝叶斯优化：使用优化算法，反复迭代多次，最终返回目标函数最小的参数组合
max_evals = 3  # 迭代的轮数
import time
start = time.time()
best = fmin(fn=objective, space=space, algo=algo, max_evals=max_evals)
train_time = time.time() - start
print(best)
print('training time is {:.4f} seconds'.format(train_time))

# 查看最优参数的测试集表现
def best_test(best):
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = best['max_depth'] + 3

    best_model = xgb.XGBClassifier(**best)
    best_model.fit(X_train, y_train)
    pre_y = best_model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, pre_y)
    print('test auc: {}'.format(auc))
best_test(best)