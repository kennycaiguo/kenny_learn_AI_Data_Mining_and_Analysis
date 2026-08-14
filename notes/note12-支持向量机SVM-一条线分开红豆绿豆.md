# 支持向量机SVM

## 通过一个案例切入：区分红豆和绿豆案例

![image-20260814134305465](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814134305465.png)

## SVM算法原理

![image-20260814134628217](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814134628217.png)

## SVM的一些相关概念

### 超平面

![image-20260814135104050](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814135104050.png)

### 支持向量

![image-20260814135309954](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814135309954.png)

![image-20260814135352562](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814135352562.png)

![image-20260814135424620](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814135424620.png)

## SVM如何处理不清晰的边界？有软间隔和硬间隔2种处理方法

![image-20260814135941797](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814135941797.png)

![image-20260814140105117](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814140105117.png)

## 如何处理非线性可分

![image-20260814140218454](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814140218454.png)



### SVM的处理方法是：

![image-20260814140401026](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814140401026.png)

![image-20260814140520914](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814140520914.png)

### 常见的核函数有3种

![image-20260814140821297](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814140821297.png)

## SVM算法的优缺点

### SVM算法的优点

![image-20260814141153141](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814141153141.png)

### SVM算法的缺点

![image-20260814141455792](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814141455792.png)





# 演练，使用jupyter lab,新建一个svm文件夹里面有一个svmsample.ipynb,内容如下

```
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn import svm

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
clf = svm.SVC(kernel='linear')
# 模型训练
clf.fit(xtrain,ytrain)
# 模型预测
ypred = clf.predict(xtest)

# 计算score
score = clf.score(xtest,ytest,sample_weight=None)
print(f"ypredit={ypred}")
print(f"ytest={ytest}")
print(f"Accuracy:{score}")
```



### 运行效果

![image-20260814144320543](./note12-支持向量机SVM-一条线分开红豆绿豆.assets/image-20260814144320543.png)



### 问题，SVM是用来处理2分类的问题，为什么这里是3分类也能够使用？





