# 从这一节开始我们就要进入自然语言处理的学习

## 1.自然语言

![image-20260821164654999](./note22-tf-idf-一种古老但是有效的关键词提取技术.assets/image-20260821164654999.png)

## 2.tf-idf算法原理

![image-20260821181555122](./note22-tf-idf-一种古老但是有效的关键词提取技术.assets/image-20260821181555122.png)

## 3.tf-idf算法的优缺点

### ![image-20260821181712890](./note22-tf-idf-一种古老但是有效的关键词提取技术.assets/image-20260821181712890.png)



## 课堂演练

```python
import gensim.downloader as api
from gensim.corpora import Dictionary

# 加载数据
# text8的数据保存在 C:\Users\kenny\gensim-data\text8\
# 里面有一个__init__.py,需要把from smart_open import smart_open改为: from smart_open import open,否则会报错


dataset = api.load("text8") # 数据保存在
dct = Dictionary(dataset)
new_corpus = [dct.doc2bow(line) for line in dataset]
# 加载模型库
from gensim import models
# 训练模型
tfidf = models.TfidfModel(new_corpus)
# 保持模型
tfidf.save("tfidf.model")

```


```python
# 加载模型
tfidf = models.TfidfModel.load("./tfidf.model")
#利用模型得到tfidf值
tfidf_vec = []
for i in range(len(new_corpus)):
    str_tfidf = tfidf[new_corpus[i]]
    tfidf_vec.append(str_tfidf)
print(len(tfidf_vec))
```
### 输出：
    1701



```python
import jieba

seg_list = jieba.cut("这是一句话，你看切成啥",cut_all=False)
print("Default model:"+ " ".join(seg_list))
```
#### 输出
    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\kenny\AppData\Local\Temp\jieba.cache
    Loading model cost 0.431 seconds.
    Prefix dict has been built successfully.


    Default model:这是 一句 话 ， 你 看 切成 啥

## 注意

![image-20260821195819633](./note22-tf-idf-一种古老但是有效的关键词提取技术.assets/image-20260821195819633.png)

# 参考文档1

1. 介绍

TF-IDF（Term Frequency-Inverse Document Frequency，词频-逆文件频率）是一种用于文本检索与文本探勘的常用[加权技术](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=加权技术&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLliqDmnYPmioDmnK8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.XXa8L3MEbszciGZbdd1Qhe9k3JsVCPj_Z5VepVtfp5E&zhida_source=entity)。TF-IDF是一种[统计方法](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=统计方法&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLnu5_orqHmlrnms5UiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.I-496rwJt4dhlFJ5oE56fb0mOfv0oH8Zj-mjM5xJ060&zhida_source=entity)，用以评估一字词对于一个文件集或一个[语料库](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=语料库&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLor63mlpnlupMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.PAeIrt4Jz9To1uuszg6yroRWXZJHXTnYwuZF3gDSkvU&zhida_source=entity)中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。这种计算方式能有效避免常用词对关键词的影响，提高了关键词与文章之间的相关性。

2. 作用与目的

TF-IDF广泛应用于[自然语言处理](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=自然语言处理&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLoh6rnhLbor63oqIDlpITnkIYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.WERhu_1bCqCKw-8bAWngD-TceF-GiRLsiQUqFAdwxLE&zhida_source=entity)和信息检索领域的各种任务，包括关键词提取、文本分类、[文本相似度计算](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=文本相似度计算&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLmlofmnKznm7jkvLzluqborqHnrpciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.o0GBH0sHXYIr5cmRKniT9XoykQucjCIH5cV_3K73-cI&zhida_source=entity)等4。通过计算文章中各个词的TF-IDF，由大到小排序，排在最前面的几个词，就是该文章的关键词。这样，TF-IDF可以帮助我们从大量文本数据中提取出有价值的信息，从而进行更深入的分析和研究。

3. 实现原理

TF-IDF分为两部分：TF和IDF。TF (Term Frequency, 词频) 表示词条在文本中出现的频率，这个数字通常会被归一化 (一般是词频除以文章总词数)，以防止它偏向长的文件。IDF (Inverse Document Frequency, [逆文件频率](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=2&q=逆文件频率&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLpgIbmlofku7bpopHnjociLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.iB7vVqVuPs7GKB5ICEVroNUluRgU4_TWcW7eYBqCRic&zhida_source=entity))表示关键词的普遍程度。如果包含词条 i 的文档越少， IDF越大，则说明该词条具有很好的类别区分能力。TF-IDF是将TF和IDF相乘得到的[权重值](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=权重值&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLmnYPph43lgLwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.KXa3Dbq2OZ-XK2f_J1bC41nAqWbtOD5RyN9DSpPbiiE&zhida_source=entity)。TF-IDF值越大，表示该词在文档中的重要性越高5。

4. 分类

TF-IDF本身并没有明确的分类，但在实际应用中，通常会对TF和IDF进行一些调整，例如使用[平滑技术](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=平滑技术&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLlubPmu5HmioDmnK8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.IkN1f-7hRbXPe7bLQ-VxQwZghzqZcy0OOSP_uCIgWO8&zhida_source=entity)，以便更好地反映词的重要性。此外，还有一些变体和扩展，如基于[n-gram](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=1&q=n-gram&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiJuLWdyYW0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.UYgySwBoX_jwjCTZagQ-WFKrxKuWefqdAVIc-_EixxQ&zhida_source=entity)的TF-IDF，它不仅考虑单个词，还考虑词的组合；还有基于词向量的TF-IDF，它结合了[词向量](https://zhida.zhihu.com/search?content_id=170516152&content_type=Article&match_order=2&q=词向量&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA0MTksInEiOiLor43lkJHph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoxNzA1MTYxNTIsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MiwiemRfdG9rZW4iOm51bGx9.eM3yXZUQa1w4gICp-j93ecDcj9DwFvBy-ThEwdisk9A&zhida_source=entity)模型，以捕捉词的语义信息。



![img](https://pic3.zhimg.com/v2-6952e007d215496afc7b92f66a70b0f6_1440w.jpg)

# 参考文档2

## **TF-IDF算法步骤**

第一步，计算词频：

![img](https://pic4.zhimg.com/v2-281a550de928afe343c055d06371cf77_1440w.jpg)

考虑到文章有长短之分，为了便于不同文章的比较，进行"词频"标准化。

![img](https://picx.zhimg.com/v2-393435b342546a2f1736d1d755adb1cd_1440w.jpg)

第二步，计算逆文档频率：

这时，需要一个[语料库](https://zhida.zhihu.com/search?content_id=4667475&content_type=Article&match_order=1&q=语料库&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA1MjIsInEiOiLor63mlpnlupMiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo0NjY3NDc1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.RF6yjfGIzkirbLTGwIPJ824OYXxcbhpVfXQRn33DW90&zhida_source=entity)（corpus），用来模拟语言的使用环境。

![img](https://pic4.zhimg.com/v2-1d5c436e04f497544d72fec6909a3fad_1440w.jpg)

如果一个词越常见，那么分母就越大，[逆文档频率](https://zhida.zhihu.com/search?content_id=4667475&content_type=Article&match_order=4&q=逆文档频率&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA1MjIsInEiOiLpgIbmlofmoaPpopHnjociLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo0NjY3NDc1LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjQsInpkX3Rva2VuIjpudWxsfQ.rtoIIFcRr3wSXG4E3d2LDlA7IMuJOvRTPOq8OD0igWc&zhida_source=entity)就越小越接近0。分母之所以要加1，是为了避免分母为0（即所有文档都不包含该词）。log表示对得到的值取对数。

第三步，计算TF-IDF：

![img](https://pic3.zhimg.com/v2-5560a4b2efa3330021b8b2ef13a471fe_1440w.jpg)

可以看到，TF-IDF与一个词在文档中的出现次数成正比，与该词在整个语言中的出现次数成反比。所以，自动提取关键词的算法就很清楚了，就是**计算出文档的每个词的TF-IDF值，然后按降序排列，取排在最前面的几个词。**

## **优缺点**

TF-IDF的优点是简单快速，而且容易理解。缺点是有时候用**词频**来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。如果要体现词的上下文结构，那么你可能需要使用[word2vec](https://zhida.zhihu.com/search?content_id=4667475&content_type=Article&match_order=1&q=word2vec&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODc1MjA1MjIsInEiOiJ3b3JkMnZlYyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjQ2Njc0NzUsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.no959pm8EiMCEhdx2JFGkK18lXXhuBVMHdqCQIcPI7M&zhida_source=entity)算法来支持。

## **示例代码**

![img](https://pic2.zhimg.com/v2-e5cde92e7c4fdee6c6eb034e023e8c1b_1440w.jpg)

# 参考文档3：

##  https://zhuanlan.zhihu.com/p/44608578

## 源码仓库：https://github.com/Jasonnor/tf-idf-python

# 参考文档4

#### NLP 入门实战：基于 jieba 分词与 TF-IDF 实现文本分类

文本分类是自然语言处理（NLP）的基础任务，旨在将文本自动分配到预定义的类别中。本实战教程将指导你使用 Python 的 `jieba` 库进行中文分词，并结合 TF-IDF（Term Frequency-Inverse Document Frequency）特征提取方法，实现一个简单的文本分类器。我们将以新闻分类为例（例如区分体育、科技等类别），逐步讲解过程。整个项目基于 scikit-learn 库实现，确保代码简洁高效。

##### 核心概念简介

- **jieba 分词**：用于中文文本的分词处理，将句子切分为词语序列。

- TF-IDF

  ：一种统计方法，用于评估词语在文档中的重要程度。公式为： $$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) $$ 其中：

  - $ \text{TF}(t, d) $ 表示词 $t$ 在文档 $d$ 中的频率。
  - $ \text{IDF}(t) = \log \left( \frac{N}{n_t} \right) $，$N$ 是总文档数，$n_t$ 是包含词 $t$ 的文档数。

- **文本分类**：使用[机器学习](https://link.csdn.net/?target=https%3A%2F%2Fedu.csdn.net%2Flearn%2F37264%2F576833%3Futm_source%3D2019755004)算法（如朴素贝叶斯）训练[模型](https://link.csdn.net/?target=https%3A%2F%2Ftaotoken.net%2F%3Fdc%3Ddc0x25b1yyscw2%26utm_source%3Dtt_distributor)，预测文本类别。

##### 实现步骤

以下步骤将帮助你逐步构建和训练分类器。我们使用朴素贝叶斯分类器作为示例，因其简单高效，适合入门。

1. **安装必要库**
   确保已安装以下 Python 库（通过 pip 安装）：

   ```mipsasm
   pip install jieba scikit-learn
   ```

   

2. **准备数据集**
   假设我们有一个简单的文本数据集，包含文本内容和标签（例如：0 表示体育，1 表示科技）。你可以使用自定义数据或公开数据集（如 THUCNews 子集）。

   - 示例数据格式：列表形式，`texts` 存储文本，`labels` 存储类别标签。

3. **分词处理**
   使用 `jieba` 对每个文本进行分词，将中文句子转换为词语列表。

   - 关键函数：`jieba.cut`，并转换为空格分隔的字符串，便于后续处理。

4. **构建 TF-IDF 特征向量**
   使用 scikit-learn 的 `TfidfVectorizer` 将分词后的文本转换为 TF-IDF 特征矩阵。

   - 参数说明：`max_features` 可限制特征维度，避免维度爆炸。

5. **分割训练集和测试集**
   将数据分为训练集和测试集，比例通常为 8:2。

6. **训练分类模型**
   使用朴素贝叶斯分类器（`MultinomialNB`）进行训练。

7. **评估模型性能**
   计算准确率等指标。

##### 完整代码示例

以下 Python 代码实现了上述所有步骤。代码中使用了 `jieba` 分词和 scikit-learn 的 TF-IDF 功能。

```python
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 步骤1: 准备示例数据集（实际中替换为你的数据）
texts = [
    "中国男篮在亚运会夺冠",  # 标签0: 体育
    "华为发布新款5G手机",   # 标签1: 科技
    "足球世界杯决赛精彩纷呈",
    "人工智能技术取得新突破"
]
labels = [0, 1, 0, 1]  # 0:体育, 1:科技

# 步骤2: 使用jieba进行分词
def chinese_tokenizer(text):
    words = jieba.cut(text)  # 分词
    return " ".join(words)  # 转换为空格分隔字符串

tokenized_texts = [chinese_tokenizer(text) for text in texts]

# 步骤3: 构建TF-IDF特征向量
vectorizer = TfidfVectorizer(max_features=1000)  # 限制特征维度
tfidf_matrix = vectorizer.fit_transform(tokenized_texts)

# 步骤4: 分割数据集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels, test_size=0.2, random_state=42
)

# 步骤5: 训练朴素贝叶斯分类器
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 步骤6: 预测并评估
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 示例预测新文本
new_text = "科学家发现量子计算新进展"
tokenized_new_text = chinese_tokenizer(new_text)
new_tfidf = vectorizer.transform([tokenized_new_text])
predicted_label = classifier.predict(new_tfidf)
print(f"新文本预测类别: {'科技' if predicted_label[0] == 1 else '体育'}")
```



##### 运行说明

- **输入**：替换 `texts` 和 `labels` 为你自己的数据集（确保数据平衡）。

- **输出**：代码会输出模型准确率和新文本的预测结果。

- 参数调整

  ：

  - 增加数据量或调整 `max_features` 可提升性能。
  - 尝试其他分类器（如 SVM：`from sklearn.svm import SVC`），替换 `MultinomialNB`。

##### 常见问题与优化建议

- **分词优化**：使用 `jieba.load_userdict("user_dict.txt")` 加载自定义词典，提高专业术语识别。
- **特征工程**：TF-IDF 可结合 N-gram（在 `TfidfVectorizer` 中设置 `ngram_range=(1,2)`）捕获上下文。
- **模型选择**：如果准确率低，尝试交叉验证或集成方法（如随机森林）。
- **扩展应用**：本方法适用于情感分析、新闻分类等任务；数据集越大，效果越好。

通过本实战，你已掌握了基于 jieba 和 TF-IDF 的文本分类基础。实际项目中，建议使用更大数据集（如搜狗新闻数据集）并优化超参数，以获得更高精度。

[# 自然语言处理](https://devpress.csdn.net/tags/629eeed4512a562a42849839)[# 分类](https://devpress.csdn.net/tags/629eeed4512a562a4284983f)

[![Logo](https://i-blog.csdnimg.cn/devpress/blog/417a1cc270c24c89a9db07f2e6d0e69d.png)](https://adg.csdn.net/)

[**智能体开发者社区**](https://adg.csdn.net/)

中国智能体开发者社区，聚焦智能体与大模型开发，提供前沿资讯、实用工具链、开源项目及行业案例。通过技术沙龙、开发者大赛等活动，促进经验交流与协作，助力开发者快速构建创新智能应用。

