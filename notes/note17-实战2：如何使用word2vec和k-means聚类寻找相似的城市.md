# 摘要

目录理解业务理解数据准备数据与模型训练训练 Word2Vec 模型训练K-means模型 总结（使用 XGB 实现酒店信息消歧）中其实没有涉及太多的代码，主要是以介绍思路为主。在这一课时中，我将提供一个较为完整的代码，带领你亲自实践一下。理解业务在旅行场景下，城市——我们通常称为目的地，是一个很重要的信息。根据用户对于目的地的偏好，我们既可以把目的地作为一个特征用于推荐系统中，也可以把目的地...





# 【数据挖掘实战】—使用 word2vec 和 k-mean 聚类寻找相似城市



------

（使用 XGB 实现酒店信息消歧）中其实没有涉及太多的代码，主要是以介绍思路为主。在这一课时中，我将提供一个较为完整的代码，带领你亲自实践一下。

# 理解业务

在旅行场景下，城市——我们通常称为目的地，是一个很重要的信息。根据用户对于目的地的偏好，我们既可以把目的地作为一个特征用于推荐系统中，也可以把目的地当作一个被推荐的信息直接推荐给用户。所以，我们有一个需求，就是把相似的目的地整理出来，然后可以通过这些相似目的地做相关推荐，或者是相关目的地的推荐。

# 理解数据

可以想到，这是一个比较典型的**聚类问题**，我们只要能够把相似的城市按照一定的相关性聚在一起，就可以完成我们的需求，当然具体的效果要根据结果不断地进行调整。那我们就来看一下我们的数据。

思来想去我们只有很多目的地的名字，但这些目的地并没有什么统一标准的特征可以给我们做向量，那么该怎么去给这些**目的地计算相关性**呢？

![img](https://img-blog.csdnimg.cn/344950d8a3f14278ae078cd115ed7d95.png)

这时不禁想到，我们**有很多用户写过游记**，这些游记里总会出现各种各样的目的地名字，对于相似的目的地，那用户所写的内容也会有一定的相似性，不管是地理位置接近，还是消费价位类似，或者是可以玩的内容存在一定的相似性。

总之，我们可以靠这些内容把这些城市的名字关联起来，而且不同于结构化的信息，游记是用户自己来写的内容，里面对于目的地的认知也是用户的认知，所以如果我们能够从中发现关联性，再应用到用户身上也是比较合理的。比如说“三亚”如果只是按客观属性来划分，那应该是“海边”，但是很多用户去三亚，除了看海本身，还有家庭出游等，这些是只能从用户的角度才会产生的认知。

这里，我们就要用到一个 **Word2Vec 算法**，它可以学习输入的文本，并输出一个词向量模型，经过 Word2Vec 算法处理之后，每一个词都会变成一个**预设长度的数值向量**。这个算法会在后面的章节进行更详细的讲解，这里我们大概知道它的功能就可以了。下面我们进入到具体代码实现的环节，看看如何训练一个这样的模型。

# 准备数据与模型训练

准备数据
我们获取所有需要用到的文本数据，在这里使用了**全量的游记文本数据**。我们首先要对数据进行清洗，去除掉异常的数据，比如内容过短、获取失败，或者是存在特殊字符、使用纯英文 / 泰语写的游记，等等。

![img](https://img-blog.csdnimg.cn/9b0c4a734f754e9498a2cd645ae4b62a.png)

 完成了这个步骤之后我们要对文本内容进行**分词**，因为我期望 Word2Vec 最终构建的向量是**词级别的**。完成分词之后，我们把数据存储在文本文件中，其中每一行是一篇内容。

接下来就要训练我们的 Word2Vec模型了。

# 训练 Word2Vec 模型

这里我们使用了一个新的算法包：**Gensim**。不知道你是否还记得我在之前介绍过这个工具包，它主要用于从原始的非结构化文本信息中，通过**无监督算法学习文本向量表达**。这里面支持 TF-IDF、LSA、LDA 和 Word2Vec 等多种算法模型。来看一下代码。

```python
import gensim #引入gensim
import os
import re
import sys
import multiprocessing #引入多线程操作
from time import time
class getSentence(object):
#初始化，获取文件路径
    def __init__(self, dirname):
        self.dirname = dirname
```

![点击并拖拽以移动](https://res.hc-cdn.com/ecology/7.11.215/v2_resources/images/global-editor/image_error.png)

文本可以存储在多个文本文件中，存放在一个文件目录下，这里构建了一个**迭代**方法，循环读取目录下的所有文件。

我这里使用的文件目录为 traindata，在 traindata 下面有 31 个语料文件，其中每个有 1G 左右，如下图所示。

![img](https://img-blog.csdnimg.cn/2e2f7d2172ff43e49ca4346793afe124.png)

```python
#构建一个迭代器
    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path):
                    try:
                        #清除异常数据，主要是去除空白符以及长度为0的内容
                        s_line = line.strip()
                        if s_line== "":
                                continue
                       #把句子拆成词
                        word_line = [word for word in s_line.split( )]
                        yield word_line
                    except Exception:
                        print("catch exception")
                        yield ""
if __name__ == '__main__':
#记录一个起始时间
    begin = time()
#获取句子迭代器
    sentences = getSentence("traindata")
#训练word2vec模型 使用句子迭代器作为语料的输入，设定的最终向量长度为200维；窗口长度为15；词的最小计数为10，词频少于10的词不会进行计算；使用并行处理
    model = gensim.models.Word2Vec(sentences,size=200,window=15,min_count=10, workers=multiprocessing.cpu_count())
#模型存储，这块记得先预先新建一个model路径，或者也可以增加一段代码来识别是否已经创建，如果没有则新建一个路径
    model.save("model/word2vec_gensim")
    model.wv.save_word2vec_format("model/word2vec_org",
                                  "model/vocabulary",
                                  binary=False)
    end = time()
#输出运算所用时间
    print ("Total procesing time: %d seconds" % (end - begin))
```

![点击并拖拽以移动](https://res.hc-cdn.com/ecology/7.11.215/v2_resources/images/global-editor/image_error.png)

在正常的情况下，我们会在 model 路径下看到几个文件。其中比较重要的两个，一个 vocabulary 是词典文件，记录了出现过的词汇以及词汇出现的次数；一个 word2vec_gensim 是生成的向量文件。

通过上面的方法，我们成功获取到了很多词汇的向量，这里我的词汇量大概有 1000w 左右。但是我们这次所需要的是寻找相似城市，所以对于那些非城市名字的词汇就没有什么价值了。

于是我们这里使用我们自己的城市词库与词汇表进行匹配，对于没有在词汇表中出现过的城市名称也没有办法计算，要把这部分剔除掉。不用担心，如果这么多的语料都没有出现过的城市也一定是没有人去过的城市。

# 训练K-means模型

下面我们就可以开始训练我们的 K-means 模型了。像我们前面用过的一样，K-means 是在 sklearn 里面的一个模块。具体步骤如下所示。

```python
import gensim
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from time import time
#加载之前已经训练好的word2vec模型
def load_model():
    model = gensim.models.Word2Vec.load('../word2vec/model/word2vec_gensim')
    return model
#加载城市名称词库
def load_filterword():
    fd = open("mddwords.txt","r")
    filterword=[]
    for line in fd.readlines():
        line=line.strip()
        filterword.append(line)
    return filterword
if __name__=="__main__":
    start = time()
#加载word2vec模型
    model = load_model()
#加载词汇表
    filterword = load_filterword()
#输出词汇表长度
    print(len(filterword))
    wordvector = []
    filterkey={}
#获取我们的城市名称词库的词向量
    for word in filterword:
        wordvector.append(model[word])
        filterkey[word]=model[word]
   #输出词汇数量
    print(len(wordvector))
#训练K-means模型，这里代码设置的聚类数为2000，最大迭代次数为100，n_jobs设置的是有多少个任务同时在跑，这样可以进行多组实验来消除初始化点带来的影响
    clf = KMeans(n_clusters=2000,max_iter=100,n_jobs=10)
    s = clf.fit_predict(wordvector)
#把模型保存下来
    joblib.dump(clf,"kmeans_mdd2000.pkl")
    labels = clf.labels_
    labellist = labels.tolist()
    print(clf.inertia_)
#把所有城市名称的聚类标签保存下来
    fp = open("label_mdd2000",'w')
    fp.write(str(labellist))
    fp.close()
#把所有城市名称保存下来，其中顺序与聚类标签顺序一致
    fp1 = open("keys_mdd2000",'w')
    for key in filterkey:
        fp1.write(key+'\n')
    print("over")
    end = time()
    print("use time")
    print(end-start)
```



![点击并拖拽以移动](https://res.hc-cdn.com/ecology/7.11.215/v2_resources/images/global-editor/image_error.png)

经过上面的步骤，我们就训练好了 **K-means** 模型，当然，经过反复尝试，最终确定的不是 2000 这个簇数量，而是使用了 100 个簇的结果。我们尝试了 50、100、200、500、1000、2000 等多个聚类的结果，经过我们最后的对比评估，100 个簇的时候效果较好，于是我们最终选择了这个模型。

下图是我从结果中抽了一些簇的 TOP 结果生成的图片，可以看到聚类的效果还是很不错的。比如右下角那一簇基本都是日本关西的城市名字，左下角基本都是川藏线上的地点。

![img](https://img-blog.csdnimg.cn/9b7a5a692ff244abbe8071bee249d64d.png)![点击并拖拽以移动](https://res.hc-cdn.com/ecology/7.11.215/v2_resources/images/global-editor/image_error.png)

有了已经训练好的模型，我们就知道了这些相似城市的名称以及它们所属的簇。接下来我们要做的，就是把这些数据存储到数据库中，并在具体的业务中进行应用了。

当然，随着时间的推移，在积累了一段时间的数据之后，我们还要对模型进行**重新迭代**，以期望获得更好的结果。

![img](https://img-blog.csdnimg.cn/1c727c8b6128446f87c7b02d01ea80aa.png)

#  总结

在这一节实践课程中，我着重介绍了整个模型训练环节的代码，其中主要写了两段代码，分别训练了 Word2Vec 模型和 K-means 模型。

在数据缺少标注的时候，聚类算法是十分常用的，它可以帮助我们了解数据情况。当然，聚类方法也存在一些局限，还需要在日常的工作中多加练习，不断积累自己的经验。

# 这个案例的数据搞不到，根本无法学习。我们还是学习下面的文本聚类吧



# 扩展：下载gensim训练数据

你可以通过 Python 代码或命令行直接下载 Gensim 内置的训练数据集和预训练模型。最常用的测试语料如 `text8` 或 `glove-wiki-gigaword-100`，默认会自动保存在电脑的 `~/gensim-data` 目录中。 [[1](https://github.com/piskvorky/gensim-data), [2](https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html)]

命令行下载

在终端中运行以下命令来下载指定的数据集（例如 `text8`）： [[1](https://radimrehurek.com/gensim/downloader.html)]

- `python -m gensim.downloader --download text8`

Python 代码下载

在你的 Python 脚本中导入 `gensim.downloader` 模块加载或下载数据： [[1](https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html)]

python

```
import gensim.downloader as api

# 自动下载并加载数据
dataset = api.load("text8")
```

# 扩展1，gensim 数据下载

## https://sourceforge.net/projects/gensim-data.mirror/files/



# 扩展2，这里有一个对旅游游记文本进行聚类的GitHub仓库

## https://github.com/Edward1Chou/textClustering



# 扩展3： 文本聚类（Kmeans、DBSCAN、LDA、Single-pass）

## https://github.com/murray-z/text_clustering/tree/master

# 扩展4 ：Learn how to process, classify, cluster, summarize, understand syntax, semantics and sentiment of text data with the power of Python!

## https://github.com/dipanjanS/text-analytics-with-python
