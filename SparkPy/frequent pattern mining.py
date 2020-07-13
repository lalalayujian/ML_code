"""
频繁模式挖掘
频繁项集挖掘：FP-Tree
频繁序列挖掘：PreSpan
"""
from pyspark import SparkContext
from pyspark.mllib.fpm import PrefixSpan, FPGrowth


# ======================频繁项集挖掘=========================
sc = SparkContext('local', 'test')
data = [['A', 'B'], ['A','C'], ['B','F','E'], ['A','B','C'], ['B','E','D']]
rdd = sc.parallelize(data, 2)

# 支持度阈值为0.3
model = FPGrowth.train(rdd, 0.3, 2)
freqItemsets = sorted(model.freqItemsets().collect())
print(freqItemsets)


# ======================频繁序列挖掘=========================
# 比如每周用户的购买记录
sc = SparkContext('local', 'test')
data = [[['a'], ['a', 'b', 'c'], ['a', 'a', 'c'], ['f'], ['d', 'b']],
        [['b', 'a'], ['a', 'f'], ['f']],
        [['f'], ['f', 'd', 'c'], ['e', 'b']],
        [['f'], ['a', 'e', 'c'], ['c', 'b'], ['a'], ['f']]
        ]
rdd = sc.parallelize(data, 2)

# 支持度阈值为0.3
model = PrefixSpan.train(rdd, 0.3, 4)
freqItemsets = sorted(model.freqSequences().collect())
print(freqItemsets)
