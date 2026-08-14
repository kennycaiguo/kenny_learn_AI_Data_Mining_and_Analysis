# 朴素贝叶斯算法

## 以案例切入

![image-20260814123916993](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814123916993.png)

### 把数据整理成表格

![image-20260814124120858](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814124120858.png)

![image-20260814124319536](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814124319536.png)

#### 此时不延误的概率大于延误的概览，不需要买保险

## 朴素贝叶斯算法的原理以及优缺点

### 算法原理

### 上面这个案例就是贝叶斯概率公式的应用

![image-20260814124607943](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814124607943.png)



### 转化为通俗的方式

![image-20260814124854334](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814124854334.png)

### 把上面的案例的数据导入公式，就会得到下面的表达式

![image-20260814125029942](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814125029942.png)

### 也可以推导出

![image-20260814125155019](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814125155019.png)

![image-20260814125546726](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814125546726.png)

### 如何处理连续值？

![image-20260814130140816](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814130140816.png)

### 关于平滑

![image-20260814130233416](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814130233416.png)

### 朴素贝叶斯算法的优点

![image-20260814131110984](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814131110984.png)



### 朴素贝叶斯算法的缺点

![image-20260814131726447](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814131726447.png)







## 项目演练，使用jupyter lab来学习

## 1.在kenny_learn_AI_Data_Mining_and_Analysis\codes文件夹里面新建一个naivebayes文件夹，在里面新建一个nbeiyes.ipynb文件，内容如下

```
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

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
clf = GaussianNB()
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



### 运行效果：

![image-20260814132003203](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814132003203.png)

## 朴素贝叶斯算法的扩展

![image-20260814132202107](./note11-朴素贝叶斯-算一算你要不要买延误险.assets/image-20260814132202107.png)

### 半朴素贝叶斯算法参考文档1： https://zhuanlan.zhihu.com/p/350772160

### 半朴素贝叶斯算法参考文档2：https://www.cnblogs.com/wang_yb/p/18859897

### 半朴素贝叶斯算法参考文档3：https://zhuanlan.zhihu.com/p/518617685

### AODE算法参考文档1：https://developer.cloud.tencent.com/article/2647352

### AODE算法参考文档2：https://zhuanlan.zhihu.com/p/377045404

### AODE算法参考文档3：https://www.geeksforgeeks.org/machine-learning/averaged-one-dependence-estimators-aode/
