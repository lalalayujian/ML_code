"""
    用朴素贝叶斯做文本分类
    预处理（分词+去停用词+词袋模型/）+朴素贝叶斯
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

samples = ["Machine learning is fascinating, it is wonderful",
           "Machine learning is a sensational techonology",
           "Elsa is a popular character"]

# 词频向量，有个问题，词频高不一定有词义，比如is
countv = CountVectorizer()
X = countv.fit_transform(samples)  # 稀疏矩阵
print('词频向量：\n', pd.DataFrame(X.toarray(), columns=countv.get_feature_names()))

# tf-idf向量，词频逆文档频率，词频/文档中出现词的文档数。可降低词频高且在所有文档中常出现的词的权重
tfv = TfidfVectorizer()
X = tfv.fit_transform(samples)
print('tf-idf向量：\n', pd.DataFrame(X.toarray(), columns=tfv.get_feature_names()))


print('=======新闻数据分类贝叶斯========')
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# 初次使用这个数据集的时候，会在实例化的时候开始下载
# data = fetch_20newsgroups()
# print(data.target_names)

categories = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "talk.politics.mideast"]
train = fetch_20newsgroups(subset="train",categories = categories)
test = fetch_20newsgroups(subset="test",categories = categories)
# 查看数据是否平衡
print(pd.Series(train.target).value_counts())

Xtrain = train.data
Xtest = test.data
Ytrain = train.target
Ytest = test.target

tfidf = TfidfVectorizer().fit(Xtrain)
Xtrain_ = tfidf.transform(Xtrain)
Xtest_ = tfidf.transform(Xtest)
tosee = pd.DataFrame(Xtrain_.toarray(),columns=tfidf.get_feature_names())
print(tosee)

# 建模，高斯朴素贝叶斯不接受稀疏矩阵
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB

name = ["Multinomial","Bernoulli","Complement"]
models = [MultinomialNB(),BernoulliNB(),ComplementNB()]

for name, clf in zip(name,models):
    clf.fit(Xtrain,Ytrain)
    y_pred = clf.predict(Xtest)
    proba = clf.predict_proba(Xtest)[:,1]
    score = clf.score(Xtest,Ytest)

    print(name)
    print("\tAccuracy:{:.3f}".format(score))
# 还可以试试概率校准
