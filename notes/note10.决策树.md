# 决策树：女神使用的约会决策

## 1.从一个例子出发，看看女神是怎样决定要不要和某人约会

### 女神约会决策图

![image-20260813151742037](./note10.决策树.assets/image-20260813151742037.png)

### 决策树执行流程

![image-20260813152023430](./note10.决策树.assets/image-20260813152023430.png)

### 假如女神收到3个人的约会邀请，他们的条件如下表

![image-20260813152115360](./note10.决策树.assets/image-20260813152115360.png)

### 我们来看看决策的结果

#### 第一个人的结果是：不见

![image-20260813152337071](./note10.决策树.assets/image-20260813152337071.png)

#### 第二个人的结果是：见

![image-20260813152504136](./note10.决策树.assets/image-20260813152504136.png)

#### 第三个人的结果是：见

![image-20260813152635019](./note10.决策树.assets/image-20260813152635019.png)



### 问题：如何选择合适的根节点？下一次又该选择哪个特征作为节点？

#### 使用信息增益法

![image-20260813152921767](./note10.决策树.assets/image-20260813152921767.png)



## 2.分析算法原理，思路形成的过程

### 理想情况

![image-20260813153030188](./note10.决策树.assets/image-20260813153030188.png)

### 实际情况.

![image-20260813153125799](./note10.决策树.assets/image-20260813153125799.png)

### 几个版本的决策树的比较

![image-20260813153239304](./note10.决策树.assets/image-20260813153239304.png)

#### 我们主要使用最后一种

### 决策树的优点

![image-20260813153714589](./note10.决策树.assets/image-20260813153714589.png)

### 决策树的缺点

![image-20260813153953908](./note10.决策树.assets/image-20260813153953908.png)

### 关于剪枝，有预剪枝和后剪枝，预剪枝的效果不好。

![image-20260813154233565](./note10.决策树.assets/image-20260813154233565.png)

## 小案例，还是iris案例

```
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
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
tree = DecisionTreeClassifier(max_depth=4)
# 模型训练
tree.fit(xtrain,ytrain)
# 模型预测
ypred = tree.predict(xtest)

probilty = tree.predict_proba(xtest)

# 计算score
score = tree.score(xtest,ytest,sample_weight=None)
print(f"ypredit={ypred}")
print(f"ytest={ytest}")
print(f"Accuracy:{score}")



```



### 为了方便绘图，我们使用jupyter lab来开发，先安装一个pydotplus

```
pip install pydotplus
```

### 然后添加绘图功能代码

```
from IPython.display import Image
from sklearn import tree
# dot是一个程序化生成流程图的简单语言
import pydotplus

dot_data = tree.export_graphviz(dt,out_file=None,feature_names=iris.feature_names,
                                class_names=iris.target_names,filled=True,rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```



### 运行程序就会得到一个图，具体可以运行dtreedemo.ipynb查看结果，和下面的图形类似

![image-20260813160809987](./note10.决策树.assets/image-20260813160809987.png)



## 3.扩展：由此衍生的高级版本

![image-20260813161041275](./note10.决策树.assets/image-20260813161041275.png)

# 思考，你对否对决策树的实现细节中还有什么疑惑？

