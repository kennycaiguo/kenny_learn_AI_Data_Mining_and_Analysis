from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 设置随机种子
np.random.seed(0)

iris = load_iris()
x,y = iris.data,iris.target

# iris数据集有150条数据，我们可以用140条作为训练集，10条作为测试集
randomarr = np.random.permutation(len(x))
# train set
xtrain = x[randomarr[:-10]]
ytrain = y[randomarr[:-10]]
# test_set
xtest = x[randomarr[-10:]]
ytest = y[randomarr[-10:]]

# 创建分类器对象
knn = KNeighborsClassifier()
# 模型训练
knn.fit(xtrain,ytrain)
# 模型预测
ypred = knn.predict(xtest)

probilty = knn.predict_proba(xtest)
# 计算最后一个测试样本距离最近的5个点
neighborpoints = knn.kneighbors([xtest[-1]],5)
# 计算score
score = knn.score(xtest,ytest,sample_weight=None)
print(f"ypredit={ypred}")
print(f"ytest={ytest}")
print(f"Accuracy:{score}")


