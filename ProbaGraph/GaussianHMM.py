"""
隐马尔科夫模型：著名的贝叶斯网(有向无环图表示变量间相关关系)
两种变量：隐藏状态、观察状态
模型有三组参数决定：隐藏状态转移概率矩阵、观察概率矩阵、隐藏状态初始分布
三种问题：给定模型，计算观察序列的概率；给定观察序列，学习模型参数；给定模型和观察序列，预测隐藏状态序列
"""


import numpy as np
from hmmlearn import hmm


"=================GaussianHMM假设观测状态是连续值，符合高斯分布====================="

# startprob = np.array([0.6, 0.3, 0.1, 0.0])
#
# transmat = np.array([[0.7, 0.2, 0.0, 0.1],
#                      [0.3, 0.5, 0.2, 0.0],
#                      [0.0, 0.3, 0.5, 0.2],
#                      [0.2, 0.0, 0.2, 0.6]])
#
# # 每个隐藏状态对应的观察服从一个二维高斯分布
# means = np.array([[0.0,  0.0],
#                   [0.0, 11.0],
#                   [9.0, 10.0],
#                   [11.0, -1.0]])  # 均值矩阵：4个2维高斯分布的均值向量
# covars = .5 * np.tile(np.identity(2), (4, 1, 1))  # 协方差张量：4个2维高斯分布的协方差矩阵
#
# model = hmm.GaussianHMM(n_components=4, covariance_type="full")
# model.startprob_ = startprob
# model.transmat_ = transmat
# model.means_ = means
# model.covars_ = covars
#
# print('===========问题一：计算观察序列的出现概率===========')
# seen = np.array([[2, 3.5], [3.1, 9.5]])
# print(model.score(seen))
#
# print('==============问题三，解码隐藏状态=================')
# state_sequence = model.predict(seen)
# print('state_sequence:', state_sequence)

print('===========问题二：给定观察序列，学习模型参数============')
print('分析股价隐藏状态')
import tushare as ts
import matplotlib.pyplot as plt


data = ts.get_hist_data('600848', start='2010-01-01', end='2019-12-31')
print(data.columns)
close_v = data['close'].values  # 当日收盘价格
volume = data['volume'].values  # 当日交易量
dates = range(len(data))
fig = plt.figure()
plt.plot(close_v)
plt.show()

diff = np.diff(close_v)
diff = np.reshape(diff, (-1, 1))  # 训练数据需要二维
model = hmm.GaussianHMM(n_components=2, n_iter=100)
model.fit(diff)
hidden_states = model.predict(diff)

fig2 = plt.figure()
for j in range(len(close_v)-1):
    for i in range(model.n_components):
        if hidden_states[j] == i:
            plt.plot([dates[j],dates[j+1]],[close_v[j],close_v[j+1]],color=['r', 'g'][i])
plt.show()

"GMMHMM假设观测状态是连续值，符合混合高斯分布"
# 每个隐藏状态对应的观察服从一个混合高斯分布