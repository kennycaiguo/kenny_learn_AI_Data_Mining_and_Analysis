# 参考文档1

# 线性回归（Linear Regression）和逻辑回归（Logistic Regression）

线性回归用于拟合数据，通过最小化平方误差函数找到最佳模型参数。逻辑回归则是用于分类问题，通过sigmoid函数转换输出并利用交叉熵作为代价函数。两者都采用梯度下降法进行参数优化，但逻辑回归的代价函数是非凸的，需要特殊处理。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2197aa78570172ccc046afc897f83e2f.png#pic_center)
先举两个简单的例子，看上面的图片。

线性回归主要功能是**拟合数据**。
逻辑回归主要功能是**区分数据**，找到决策边界。

线性回归的代价函数常用**平方误差函数**。
逻辑回归的代价函数常用**交叉熵**。

参数优化的方法都是常用**梯度下降**。

------

## 1**.线性回归**

在介绍逻辑回归之前，先用线性回归来热热身。线性回归几乎是最简单的模型了，它假设因变量和自变量之间是线性关系的，一条直线简单明了。

在有监督（有标签的）学习中，我们会有一份数据集，由一列观测（**y**，即因变量）和多列特征（**X**，即自变量）组成 。线性回归的目的就是找到和样本拟合程度最佳的[线性模型](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=线性模型&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLnur_mgKfmqKHlnosiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Jbl--mcDWWalSmkngEXH9AgCy9j-ynTRysUshle6-as&zhida_source=entity)（or式子，方程 whatever），在寻找过程中需要确定系数**β**和干扰项**ε**（干扰项的作用是捕获除了X之外所有影响y的其他因素）。

直接上公式吧，有“看公式会发困病”的同学可以直接跳过到 ***加粗斜体黑字\*** 噢 (●ˇ∀ˇ●)~：

**y**是一列有n个观测值的观测变量，或者直接说因变量以便于理解，

**X**是由多列特征组成的特征空间，假设有p列特征，简单理解就是有p个[自变量](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=3&q=自变量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLoh6rlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjMsInpkX3Rva2VuIjpudWxsfQ.-FI0pCbF4laNmTk7wAB-hzBIaRDd0QO_vzUIg8G16wM&zhida_source=entity)，每个特征都有n个值，这与y是对应的：

![image-20260820191627378](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820191627378.png)

**β**是[系数向量](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=系数向量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLns7vmlbDlkJHph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.p6U7dWBhcjJ6mf2nVTVy3ryDZwsTwkXaN6BFzyjDJ1o&zhida_source=entity)，**ε**是干扰项（disturbance term），或称错误项（error term）：

![image-20260820191659279](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820191659279.png)

最后我们得到的第i个y（[观测值](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=2&q=观测值&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLop4LmtYvlgLwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjIsInpkX3Rva2VuIjpudWxsfQ.CZ15j7SZK0sXRsv-p1lXOTNB1ob8x-v7zLbzgO1BKyA&zhida_source=entity)）是这样的：

![image-20260820191718680](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820191718680.png)

为截距

***所以，线性回归的公式是这样子的：\***

![image-20260820191741606](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820191741606.png)

前面说过，线性回归的过程就是要**找到最优的模型来描述数据**。这里就产生了两个问题：

- 如何定义“最优”？
- 如何寻找“最优”？

想要评价一个模型的优良，就需要一个**度量标准**。对于回归问题，最常用的度量标准就是**[均方差（MSE，Mean Squared Error）](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Mean_squared_error)**，均方差是指预测值和实际值之间的平均方差。[平均方差](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=2&q=平均方差&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlubPlnYfmlrnlt64iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjIsInpkX3Rva2VuIjpudWxsfQ.mh9GIEUaOnJNMhKye79era-2DzesAVsyHV4DTp1gfQ4&zhida_source=entity)越小，说明测试值和实际值之间的差距越小，即模型性能更优。

在线性回归的式子中y和X是给定的，而β和ε是不确定的，也就是说，**找到最优的β和ε，就找到了最优的模型**。

综合以上结论，可以用如下式子描述：

![image-20260820191806684](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820191806684.png)

其中， 和 是要求的最优参数，右部是[最小化均方差](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=最小化均方差&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLmnIDlsI_ljJblnYfmlrnlt64iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.NQhUNXORYIPO_vqQEB8SLTB4V5zdJpzttyKF3kAO54A&zhida_source=entity)。

明确了我们的目标，接下来就是该如何去寻找这两个量呢？最常用的是参数估计方法是**[最小二乘法（Least Square Method）](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)**, 最小二乘法试图找到一条直线，使得样本点和直线的[欧氏距离](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/zh-hans/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E8%B7%9D%E7%A6%BB)之和最小。这个寻找的过程简单描述就是：根据[凸函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=凸函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlh7jlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.DyNnyBKi_RFWnPRZwioAV99eHYFD-Bw1qfgU8797oek&zhida_source=entity)的性质，求其关于β和ε的[二阶导](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=二阶导&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLkuozpmLblr7wiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.-BqaS5W2RNyH1HVdZzcQTjCdVaZEEUq9KwZH0AvUs9Y&zhida_source=entity)的零点。





( •̀ ω •́ )✧终于讲完了理论知识，下面展示一个使用[python3](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=python3&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiJweXRob24zIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6NzgwOTI5MiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.K_ELX8wc2BQCd1KKD96w9fTX4aHP5ObfYagVeWbT5Qs&zhida_source=entity)编写的简单例子。

```python
from numpy.random import randint # 导入numpy库中的随机数生成函数
import matplotlib.pyplot as plt  # 导入绘图库

X = np.array([x for x in range(1, 101)]).reshape(-1, 1) # X = 1,2,...,n
y = np.array(list(map(lambda x: 2 * x + randint(-30, 30), X))) # y=2X+r, r=random(-30,+30)
y_true = np.array(list(map(lambda x: 2 * x, X))) # y=2x
plt.plot(X, y_true, c='b') # 绘制线图
plt.scatter(X, y, alpha=0.7, s=60) # 绘制散点图
plt.title('Random Scatter') 
plt.show()
```

![img](https://pic3.zhimg.com/v2-ce26771eb9dbe331332170da320e605a_1440w.jpg)

即使去掉标准的y=2x直线图像，也可以很明显地看出y和X具有线性关系（这里的X为了简单起见只包含一个自变量，即X[特征空间](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=2&q=特征空间&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLnibnlvoHnqbrpl7QiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjIsInpkX3Rva2VuIjpudWxsfQ.cYY03kXoVWbxK7vCTWZ922Lk54TXv6NNsWV173w0ecc&zhida_source=entity)只有一维/列），但是由于我们加入了噪声（随机数），所以仅仅靠观察的话找不到一条确切的直线能够描述y和X的线性关系。

```python
# 续上面的代码
from sklearn.linear_model import LinearRegression # 导入sklearn库中的线性回归模块
lr = LinearRegression() # 定义一个线性回归模型
lr.fit(X, y) # 将模型拟合到数据上
y_pred = lr.predict(X) # 使用模型预测数据（方便起见就不划分训练集和测试集了）
plt.scatter(X, y, alpha=0.5, s=50)
plt.plot(X, y_pred, c='r')
plt.plot(X, y_true, c='b')
plt.title('Linear Regression')
plt.show()
```

![img](https://pic4.zhimg.com/v2-670ac278200c8d408741db4218b20265_1440w.jpg)

图中的蓝线是y=2x，红线是回归之后得到的[拟合直线](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=拟合直线&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLmi5_lkIjnm7Tnur8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.jLNJ6CYkHCu4I9llz1W-udHzIhp9oKJrpsC8uQkNx1w&zhida_source=entity)，由于噪声比较多，所以拟合程度没有非常理想，但也足以描述观测量y和[回归量](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=回归量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlm57lvZLph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.aiwrRAH9Irz0JoPu1uLOx_u-l-3Bux1ksqQwlO8-0iE&zhida_source=entity)X之间的线性关系了。



## **2.从线性回归推广到广义线性回归**

上面我们得到了线性回归模型的数学原型，在数学上一个特例经常都是归属于一个更普遍或更一般的原型。让我们思考下面这两个回归模型：

![image-20260820192537753](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820192537753.png)

左边是我们之前得到的线性回归模型，右边是对数线性回归模型（log-Linear Regression）。从等式的形式来看，对数线性回归与线性回归区别仅仅在于等式左部，形式依旧是线性回归，但实质上是完成了输入空间X到输出空间y的**[非线性映射](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=非线性映射&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLpnZ7nur_mgKfmmKDlsIQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Bx7QSc8G7iCUNRCAEfR9O5dLUEfLRiH2Lval9l5kX18&zhida_source=entity)**。这里的[对数函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=对数函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlr7nmlbDlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.w4v8yKN9TmKYcAPbwuuGbnRm6DG0y1cCaxnJr-rEL5k&zhida_source=entity)ln(·)，将线性回归模型和真实观测**联系**起来。通俗地说，原本线性回归模型无法描述的非线性y，套上了一个[非线性函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=非线性函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLpnZ7nur_mgKflh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.XuF5b-OMPFBzvmWBHeBpYdg5Q3s3kfEF_N8WnlABQzw&zhida_source=entity)ln(·)，就可以描述对数形式的y了。

![image-20260820192600793](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820192600793.png)

将以上两个式子综合，写成更一般的形式就是广义线性回归模型（GLM，Generalized Linear Model）了。这里的g(·)，即ln(·)，是一个单调可微函数，称为**联系函数（Link Function）**。显然，前面的线性回归和[对数回归](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=对数回归&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlr7nmlbDlm57lvZIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.WrUanjgrzh6e94ztzEMRH5LrwRT_ufFo38J9LnWBo6k&zhida_source=entity)都是广义线性回归的特例，根据联系函数的不同，以不同的方式映射，可以是对数，可以是指数，也可以是其他更复杂的函数。



## **3.逻辑回归**

经过上面的铺垫，终于可以愉快地谈谈逻辑回归了✪ ω ✪

当我们想将线性回归应用到**分类问题**中该怎么办呢？比如二分类问题，将X对应的y分为类别0和类别1。我们知道，线性回归本身的输出是连续的，也就是说要将连续的值分为[离散](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=离散&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLnprvmlaMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.-LsiuOVWVhdSbwRqQ98nFvkU12E6-TlAmJoPnTMPPW4&zhida_source=entity)的0和1。答案很容易想到，找到一个联系函数，将X映射到y∈{0，1}。

可能大家立马会想到单位阶跃函数（unit-step Function），函数图像如下：

![img](https://pic1.zhimg.com/v2-d4bacc9ef3dedccbc3cebfb2820249c0_1440w.jpg)

[函数原型](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=函数原型&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlh73mlbDljp_lnosiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.20AHbhPT7SPGbDHm-xjIfn1ouRHOi1Qtbhz0LwmCob8&zhida_source=entity)如下：

![image-20260820192640992](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820192640992.png)

[单位阶跃函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=2&q=单位阶跃函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLljZXkvY3pmLbot4Plh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjIsInpkX3Rva2VuIjpudWxsfQ.zv9jokVGGJN0uvKQu4aC3yLd7F0Z2eUOmznm-qkC7HY&zhida_source=entity)的确直接明了，小于0为类别0，大于0为类别1，等于0则皆可。但是有一个原则性的问题，我们需要的联系函数，必须是一个单调可微的函数，也就是说必须是连续的。（关于连续和[可微](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=3&q=可微&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlj6_lvq4iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjMsInpkX3Rva2VuIjpudWxsfQ.SD5-FVtolJXYZm19VnmRMm7vDi7aQ9JBfW1vio0Eve8&zhida_source=entity)的概念，忘了的同学赶紧回去补[高数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=高数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLpq5jmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.fkmH4eH4phNci-C7PwJGH1XzL_kEdp7vHudkyZP2qBM&zhida_source=entity)吧╮（╯＿╰）╭）。

这里写给出一个结论，**逻辑回归使用的联系函数是Sigmoid函数（S形函数）中的最佳代表，即[对数几率函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=对数几率函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlr7nmlbDlh6Dnjoflh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.yGl0kxoX-vRvKAC9PUDpoUI7hVpOGJ0AJob9L0e1dbI&zhida_source=entity)（Logistic Function）**，函数图像如下：

![img](https://pic1.zhimg.com/v2-2ca2899406f46af41d50f06924bea49e_1440w.jpg)

函数原型如下：

![image-20260820193342255](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820193342255.png)

为什么叫对数几率呢，因为它本来是长这样子的：

![image-20260820193402995](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820193402995.png)

式子中的 ![image-20260820193426113](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820193426113.png)就是所谓的几率（odds)

> 这里插入一个小知识，不想看的同学可以直接跳过噢：
> Odds（几率）和Probability（概率）之间是有区别的
> Probability是指，期望的结果/所有可能出现的结果
> Odds是指，期望的结果/不期望出现的结果
> For example: 6个白球，4个黑球
> Prob(白球)=6/10=0.6，而Odds(白球)=6/(10-6)=1.5

将对数几率函数代入到之前得到的广义线性回归模型中，就可以得到逻辑回归的数学原型了：

![image-20260820193451999](./note18-线性回归和逻辑回归：找到一个函数取拟合数据.assets/image-20260820193451999.png)

**( •̀ ω •́ )✧所以逻辑回归也是广义线性回归中的一种以对数几率函数为联系函数的特例。**至于为什么要使用Sigmoid函数中的对数几率函数，这涉及到[伯努利分布](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=伯努利分布&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLkvK_liqrliKnliIbluIMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Ge5niJ5AKx99W2QFDzVEQT6FvdB32pA0Ug6Fv-1RVKs&zhida_source=entity)的指数族形式，最大熵理论等，这里就不展开讨论啦，数学和英语比较ok的同学可以看下这两篇详细的数学推导：

[逻辑回归与最大熵模型www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf](https://link.zhihu.com/?target=http%3A//www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf)

[指数族分布与广义线性回归blog.csdn.net/u011467621/article/details/48197943](https://link.zhihu.com/?target=https%3A//blog.csdn.net/u011467621/article/details/48197943)

得到了逻辑回归的数学原型，接下来就是找到最优参数。

与线性回归不同的是，逻辑回归由于其联系函数的选择，它的参数估计方法不再使用最小二乘法，而是**[极大似然法（Maximum Likelihood Method）](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/zh-hans/%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1)。**

最小二乘法是**最小化预测和实际之间的欧氏距离**，极大似然法的思想也是如出一辙的，但是它是通过**最大化预测属于实际的概率**来最小化预测和实际之间的“距离”。详细推导涉及凸优化理论，梯度下降法，[牛顿法](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=牛顿法&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLniZvpob_ms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Na3KOCLNFMGU4icy6bEazsWxvgCj5Y6jbh-GvxWOa9E&zhida_source=entity)等，就不展开了。



讲完了理论知识，为了防止大家发困(￣o￣) . z Z，下面同样讲个简单的例子。

```python
import matplotlib.pyplot as plt # 导入matplotlib绘图库
import seaborn as sns # 导入seaborn绘图库
  
tips = sns.load_dataset('tips') # 加载小票数据集
tips['big_tip'] = (tips['tip'] / tips['total_bill']) < 0.15 # 构造y
sns.regplot(x=tips['total_bill'], y=tips['big_tip'], logistic=True)
plt.title('Logistic Regression')
plt.show()
```

上面是一段简单的逻辑回归[可视化](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=可视化&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlj6_op4bljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Dlaj2ZlyAFzIukkJtqFWEVGZitaW2k3ppP2HSn4Ohlo&zhida_source=entity)代码

![img](https://pic4.zhimg.com/v2-fa45461df57a36f3af572921b7d8f3d3_1440w.jpg)

tips是一个seaborn内置的数据集，搜集了小票的信息，包括七个特征字段。注意这里没有类标，即没有y，所以在有监督学习的情况下，需要我们自己构造一个y。在我上面给出的代码中，y代表的含义是{0：小费超过账单的15%，1：小费没超过账单的15%}。为了方便可视化，这里只将total_bill一列作为我们的X。

![img](https://pic1.zhimg.com/v2-6f7e57aca051b53c0e14b194f8d0f774_1440w.jpg)

这就是最后绘制出来的逻辑回归图像，由于现实中的数据不总是理想化的，所以很少出现像之前展示的[sigmoid函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=sigmoid函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiJzaWdtb2lk5Ye95pWwIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6NzgwOTI5MiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.3J5Th5Lycy36XQETipsjtok8D_rZFPFjm_XEzujCPO0&zhida_source=entity)图像那么的“S”，但也能看出规律：账单较小的更有可能给出超过15%的小费，这也符合生活经验。



## **4.线性回归和逻辑回归的区别和联系**

- 线性回归和逻辑回归都是**广义线性回归模型的特例**
- 线性回归只能用于**回归问题**，逻辑回归用于**分类问题**（可由二分类推广至多分类）
- 线性回归无[联系函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=8&q=联系函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLogZTns7vlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjgsInpkX3Rva2VuIjpudWxsfQ.MRMa386h4eUEjFBj73UXHX6c3l-LYQe5D5yail89OD8&zhida_source=entity)或不起作用，逻辑回归的联系函数是**对数几率函数**，属于[Sigmoid函数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=3&q=Sigmoid函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiJTaWdtb2lk5Ye95pWwIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6NzgwOTI5MiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjozLCJ6ZF90b2tlbiI6bnVsbH0.WbEiRtsSCaY60v9-Xm7TAuri-NPG4tn2C9gAWydS5qY&zhida_source=entity)
- 线性回归使用**最小二乘法**作为参数估计方法，逻辑回归使用**极大似然法**作为参数估计方法



## **5.最后**

ヾ(￣▽￣)最后，揭晓刚开始两个问题的答案！

> 1.逻辑回归的“Logistic”应该怎么解释？
> 2.为什么逻辑回归是分类算法？

其实如果有仔细看正文的同学早就已经知道答案啦：

1.Logistic并非逻辑的意思，其语义来自Logarithm：[对数](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=13&q=对数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlr7nmlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEzLCJ6ZF90b2tlbiI6bnVsbH0.vYACkd0BBinUYzbdM9L5iCXnNaxPkGPc89K3NROGFQY&zhida_source=entity)。这更体现了Logistic Regression的本质。[周志华](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=周志华&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlkajlv5fljY4iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.AYqvyNcOtd-ebCHOWoJlb_OWDtwItXRhxK892-osZAY&zhida_source=entity)老师在其书《机器学习》中，给出了一个更恰当的中文名称：**对数几率回归**。我觉得这个翻译比起不搭边的“逻辑回归”，或者画蛇添足的“[逻辑斯谛回归](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=1&q=逻辑斯谛回归&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLpgLvovpHmlq_osJvlm57lvZIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.jR5RM_bL4KABbI_k67lStJTFsGpDex7VilqijXu81yo&zhida_source=entity)”更靠谱。

2.对数几率回归的“回归”并非针对可以应用的问题，而是来自其父级：[广义线性回归模型](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=4&q=广义线性回归模型&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlub_kuYnnur_mgKflm57lvZLmqKHlnosiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjQsInpkX3Rva2VuIjpudWxsfQ.eYCY9hELc8blqfqH8fs7_w58Z5kR2eXFROJeIXg7iN8&zhida_source=entity)。[对数几率回归](https://zhida.zhihu.com/search?content_id=7809292&content_type=Article&match_order=3&q=对数几率回归&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc0NDc0MzksInEiOiLlr7nmlbDlh6Dnjoflm57lvZIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo3ODA5MjkyLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjMsInpkX3Rva2VuIjpudWxsfQ.a7JY1GfOM78NQvjOGRvK670u_9vPs7loIDj450EFIFk&zhida_source=entity)之所以用于离散的分类而不是连续的回归，是因为它将本来连续的输出，通过对数几率函数，映射到了非线性的{0，1}空间上，所以它可以有效地解决二分类问题（甚至可推广至多分类）。

讲到这里，这篇解析就结束啦，希望我的一些见解能帮助到你

------

这是这个专栏开坑以来第一篇实际意义上的机器学习算法模型解析。看起来很简单的线性回归和对数几率回归模型，细究起来还是大有乾坤的。对于初学者来说其实不用对每一个细节，每一个推导步骤都了解清楚（毕竟数学不是所有人的特长_(:з)∠)_），只需要知道其基本原理和应用就足够了，这些细节留待以后更深入研究的时候再钻研。

还是那句话：机器学习很有趣也很有用，大家一起来学吧o((>ω< ))o！！！

# 参考文档2

## https://zhuanlan.zhihu.com/p/74874291

- 