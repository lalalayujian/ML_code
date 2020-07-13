"""
矩阵分解协同过滤做推荐
"""


from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

sc = SparkContext('local', 'testing')

user_data = sc.textFile('./ml-100k/u.data')  # 一共四列
# print(user_data.first())
# 数据是用\t分开的，划分成数组，去掉最后一列时间戳
rates = user_data.map(lambda x: x.split('\t')[0:3])
print(rates.first())

rates_data = rates.map(lambda x: Rating(int(x[0]), int(x[1]), int(x[2])))
print(rates_data.first())

# sc.setCheckpointDir('checkpoint/')
# ALS
model = ALS.train(rates_data, rank=20, iterations=5, lambda_=0.02)

print('预测用户38对物品20的评分:')
print(model.predict(38, 20))

print('预测用户38最喜欢的10个物品:')
print(model.recommendProducts(38, 10))

print('预测物品20可能最值得推荐的10个用户:')
print(model.recommendUsers(20, 10))

print('预测每个产品最值得推荐的3个用户:')
print(model.recommendUsersForProducts(3).collect())

print('预测每个用户最喜欢的3个物品')
print(model.recommendProductsForUsers(3).collect())
