# 人工神经网络

## 神经网络的背景

![image-20260814144901773](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814144901773.png)

## 人工神经网络的基础算法

### 先来看一个案例

![image-20260814145202457](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814145202457.png)

### 人的大脑的工作原理

![image-20260814145519489](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814145519489.png)

## 神经网络的算法原理

### 先来看一个最简单的神经系统



![image-20260814150044439](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814150044439.png)

### 然后我们在输入和输出层之间添加一个隐藏层，此时效果就不一样了

![image-20260814150503688](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814150503688.png)

### 原理，流程大概有四步

![image-20260814151410172](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814151410172.png)

### 关于激活函数

![image-20260814151800176](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814151800176.png)

## 神经网络算法的优缺点

### ![image-20260814151924256](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814151924256.png)

### 神经网络在信用评级和风险控制等方面不能使用



## 演练，使用jupyter lab，新建一个neuralnetwork文件夹，在里面新建一个neuralnetworkdm.ipynb文件，代码如下

```
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

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

clf = MLPClassifier(
    solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1
)
# 模型训练
clf.fit(xtrain,ytrain)
# 模型预测
ypred = clf.predict(xtest)
# 评分
score = clf.score(xtest,ytest,sample_weight=None)
print(f"ypredit={ypred}")
print(f"ytest={ytest}")
print(f"Accuracy:{score}")
print(f"layers nums:{clf.n_layers_}")

```



### 运行程序，发现效果非常差

![image-20260814153244861](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814153244861.png)

## 我们修改一下hidden_layer_size参数

![image-20260814153459981](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814153459981.png)



### 重新运行程序，效果就上来了

![image-20260814153621606](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814153621606.png)

# 扩展内容：大力出奇迹

![image-20260814153853977](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814153853977.png)

# 总结

![image-20260814154028573](./note13-人工神经网络：当前最火热的深度学习基础.assets/image-20260814154028573.png)