# Project3 for the Biomedical Information Retrieval Course
###### tags: `NCKU` `python` `生醫資訊`

## Environment
* macOS
* python3 
* Flask 2.0.2
* matplotlib 3.4.3
* nltk 3.6.5
* torch 1.10.0
* sklearn

## Requirement
* Implement **word2vec** for a set of text documents from PubMed.
* Choose one of the 2 basic neural network models to preprocess the text set from document collection.
    * **Continuous Bag of Word (CBOW)**: use a window of word to predict the middle word.
    * **Skip-gram (SG)**: use a word to predict the surrounding ones in window. Window size is not limited. Computer languages are not limited.

## Overview

### Flask & Bootstrap 5
> 參考 : https://hackmd.io/@yyyyuwen/BIR_Project2

### Word to Vector
***Word2Vec***是從大量文本中以非監督的方式學習語義的的一種模型，被大量用在NLP之中。Word2Vec是以詞向量的方式來表示語義，如果語義上有相似的單字，則在空間上距離也會很近，而 ***Embedding***是一種將單字從原先的空間映射到新的多維空間上，也就是把原先詞所在空間嵌入到一個新的空間中。
以 $f(x) = y$來看，$f()$可以視為一個空間的概念，而$x$則是***embedding***也就是表示法，$y$是我們預期的結果。
我們最常見的方式就是利用one-hot編碼建立一個詞彙表，而我訓練文檔有大約14,000個不重複的單詞，代表每一個詞彙就會是一個用0和1表示的14,000維的向量。

### CBOW & Skip-gram
![](https://i.imgur.com/q3Zh8uw.png)
CBOW和Skip-gram的model是非常相似的，主要的差異是CBOW是周圍的自在預測現在的字，而Skip-gram則是用現在的字去預測周圍的字。其中Window size是上下文的範圍(ex. Window size = 1指說取前後一個單字。)

#### Model Architecture
![](https://i.imgur.com/ecAh8Tk.png)

```python=
SkipGram_Model(
  (embeddings): Embedding(14086, 600, max_norm=1)
  (linear): Linear(in_features=600, out_features=14086, bias=True)
)
# Input Layer : 1 x 14,086
# Hidden Layer : 14,000 x 600
# Output Layer : 600 x 14,086
```

### Data pre-Processing

![](https://i.imgur.com/kFBehQA.png)

#### 1. **讀檔**
讀取4000篇.xml，取Title、Label、AbstractText
#### 2. **將文章分段轉成Sentences，取Stopword**

```python=
sentences = sent_tokenize(text)
sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in sentences]
clean_words = []
for sent in sentences:
    words = [word for word in sent.split() if not word.replace('-', '').isnumeric()]
    words = stop_word(words)
    clean_words.append(' '.join(words))
```
#### 3. **將句子切成單字**
```python=
tokens = [x.split() for x in text]
```
#### 4. **Lemmatizer**
> [Lemmatization in Python](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/) 

首先先將各個單字做詞性標註，最後再將字還原回去。
```python=
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
lemma_word = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for sentence in sentences for w in sentence]
```

#### 5. **建立詞彙表**
將各個單字建立詞彙表，並單獨標示編號。
```
{'map': 4314, 'html': 4315, 'interchange': 4316, 'vtm': 4317, 'restrictive': 4318, 'pre-analytic': 4319, 'disadvantageous': 4320, 'unidirectional': 4321, 'wiley': 4322, 'periodical': 4323, 'alternate': 4324, 'low-throughput': 4325}
```

#### 6. **建立pair**
將詞彙表的編號建立成pair，`window_size = 2`
![](https://i.imgur.com/YuPEzgo.png)

```
[(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5), (4, 6), (5, 3), (5, 4), (5, 6)]
```
### visualization
#### PCA (Principal Components Analysis) 
> [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
將word vectors降維至二維的樣子，而關聯度高的單字會聚集在一起。
Input word: covid-19
![](https://i.imgur.com/A7wdB2H.png)



## Demo
> 前面參考：https://hackmd.io/@yyyyuwen/BIR_Project2

點選Skip Gram輸入單字，會列出該單字前15關聯性的單字。
Input word: covid-19
![](https://i.imgur.com/ecTEx7d.png)


## Reference
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
* [Word2vec with PyTorch: Implementing Original Paper](https://notrocketscience.blog/word2vec-with-pytorch-implementing-original-paper/)
* [[自然語言處理] Word to Vector 實作教學](https://medium.com/royes-researchcraft/自然語言處理-2-word-to-vector-實作教學-實作篇-e2c1be2346fc)
* [Word2Vec Implementation](https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28)
* [Word2vec from scratch (Skip-gram & CBOW)](https://medium.com/@pocheng0118/word2vec-from-scratch-skip-gram-cbow-98fd17385945)
* [Skip-Gram負採樣by Pytorch](https://zhuanlan.zhihu.com/p/105955900)
* [PyTorch 實現 Skip-gram](https://zhuanlan.zhihu.com/p/275899732)
* [讓電腦聽懂人話: 直觀理解 Word2Vec 模型](https://tengyuanchang.medium.com/讓電腦聽懂人話-理解-nlp-重要技術-word2vec-的-skip-gram-模型-73d0239ad698)
* [降維與視覺化](https://ithelp.ithome.com.tw/articles/10243725)
* [Scikit-learn介紹(10)_ Principal Component Analysis](https://ithelp.ithome.com.tw/articles/10206243)
* [dict sort](https://ithelp.ithome.com.tw/articles/10222946)
* [Lemmatization和Stemming](https://ithelp.ithome.com.tw/m/articles/10214221)
* [Build your own Skip-gram Embeddings and use them in a Neural Network](https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296)
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* [Word2Vec (skip-gram model): PART 1 - Intuition.](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)
* [Implementing word2vec in PyTorch (skip-gram model)](https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb)