"""
隐马尔科夫模型：著名的贝叶斯网(有向无环图表示变量间相关关系)
两种变量：隐藏状态、观察状态
模型有三组参数决定：隐藏状态转移概率矩阵、观察概率矩阵、隐藏状态初始分布
三种问题：给定模型，计算观察序列的概率；给定观察序列，学习模型参数；给定模型和观察序列，预测隐藏状态序列
"""


import numpy as np
from hmmlearn import hmm


"=================MultinomialHMM假设观测状态是离散值====================="

states = ['box 1', 'box 2', 'box 3']
n_states = len(states)  # N=3

observations = ['red', 'white']
n_observations = len(observations)  # M=2

# 隐藏状态的初始分布
start_probability = np.array([0.2, 0.4, 0.4])

# 隐藏状态的转换概率矩阵
transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# 隐藏状态到观察状态的概率矩阵
emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

# 给定参数
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

print('===========问题一：计算观察序列的出现概率===========')
seen = np.array([[0, 1, 0]]).T
print(seen.shape)
print('seen logprob: {}'.format(model.score(seen)))

print('==============问题三，解码隐藏状态=================')
# logprob, state_sequence = model.decode(seen, algorithm='viterbi')
# print('The ball picked:', ','.join(map(lambda x: observations[x], seen[:, 0])))
# print('The hidden box:', ','.join(map(lambda x: states[x], state_sequence)))

state_sequence = model.predict(seen)
print('The ball picked:', ','.join(map(lambda x: observations[x], seen[:, 0])))
print('The hidden box:', ','.join(map(lambda x: states[x], state_sequence)))

print('===========问题二：给定观察序列，学习模型参数============')
# MultinomialHMM只支持观察数据是单维的

# 每个样本序列长度不一致，需要设置lengths控制每个样本序列的长度
X1 = np.array([[1, 1, 1]]).T  # 单个样本序列
X2 = np.array([[0, 1, 0]]).T
X3 = np.array([[1, 0, 1, 0]]).T
X4 = np.array([[1, 0, 0, 1, 1]]).T
# 将样本合并成t个时间点
X = np.concatenate([X1, X2, X3, X4], axis=0)  # (15,1)
model = hmm.MultinomialHMM(n_components=2)
model.fit(X, lengths=[2, 2, 2, 3])
state_sequence = model.predict(X1)
print('state_sequence:', state_sequence)
print(model.emissionprob_)

# 每个样本序列长度一致
X = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 0, 0]])
model = hmm.MultinomialHMM(n_components=2)
model.fit(X)
state_sequence = model.predict(X1)
print('state_sequence:', state_sequence)




