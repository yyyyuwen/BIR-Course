# Project2 for the Biomedical Information Retrieval Course
###### tags: `NCKU` `python` `生醫資訊` `Flask` `jinja2`

## Environment
* macOS
* python3 
* Flask
* nltk
* numpy
* matplotlib
* bootstrap5

## Requirement
* 實現zipf distribution（對文字的頻率的認識）
* pudmed XML文件 & Twitter json文件 
    * 取不同的數量並分析彼此差別
    * stopword
    * porter's Aigorithm
    * edit-matching

## Flask
[Flask](https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application) 主要是由[Werkzeug WSGI 工具箱](https://werkzeug.palletsprojects.com/en/2.0.x/)和[Jinja2 模板引擎](https://werkzeug.palletsprojects.com/en/2.0.x/)構成
```
Flask path
/file
/static
  |-/img
  |-/js
/templates
app.py
```
`file`儲存一些檔案處理過的資料
`static`一般是儲存image、javaScript跟css
`templates` 儲存html格式

要注意不可以寫成`flask.py`，會出錯
```python=
export FLASK_APP=app.py
flask run
```
### 與html傳遞的方法
可以使用route()來告訴FLASK，並在後面填入要載入的url位址（網路路徑）中。首頁就可以寫成 `@app.route(“/”)`。通常我會把function name跟網路路徑取相同的名稱。

### Jinja2樣版引擎
Jinja2可以將HTML頁面與後台的程式連起來，達到簡化HTML的目的。也可以方便地將參數傳至HTML並顯示出來。

#### render_template
> render_template('name.html', **lacal()) 

`**local()`是將funciton裡面的所有參數都傳到前端去，也可以使用 `name = name`這個用法選擇指定的參數傳入。而在.html檔中顯示參數的方法為`{{ name }}`。

#### 語法
Jinja2可以使用一些ifelse、for、list、set、dict這些方法。
**if else :**
```htmlembedded=
{% if path|e == '/show_text' %}
    <!-- 內容 -->
{% elif path|e == '/show_json' %}
    <!-- 內容 -->
{% else %}
    <!-- 內容 -->
{% endif %}
```

**for loop :**
```htmlembedded=
{% for word_list in new_list %}
    <!-- 內容 -->
{% endfor %}
```

**Dictionary :**
```htmlembedded=
{% for searched_word, article in word_list.items() %}

{{searched_word}} #key值
{{article}} #value值

{% endfor %}
```

**List :**
```htmlembedded=
{{list.0}}
{{list.1}}
```
> 更多用法：[python jinja2 + flask](https://hackmd.io/@shaoeChen/SJ0X-PnkG?type=view)
#### 樣板繼承方法
有時候為了不讓相同的語法重複使用(例如固定列)，我們可以用繼承的方法來簡化他。
子樣版通常會寫成
```htmlembedded=
{% extends "父樣版.html" %}
```
並注意extends一定要放在繼承的html中的第一個標籤。
在被繼承的html檔中放入想要填入子樣版的內容，會使用
```htmlembedded=
{% block content %}{% endblock %}
```
content可以是任何變數，但一定要跟子樣版的變數一樣。且同一個頁面中，content不可以重複。

#### GET & POST傳遞資料
> 參考資料：[Python Web Flask — GET、POST傳送資料](https://medium.com/seaniap/python-web-flask-get-post傳送資料-2826aeeb0e28)

當伺服器接受Request且提供對應的Response的溝通行為，是HTTP Request的一個生命週期。
這次search功能是使用POST來實作的
import方法：
`from flask import Flask, render_template, request`

在HTML檔裡頭定義POST以及設定要對應function的名稱
```htmlembedded=
<form class="d-flex" method = "POST" action = "{{url_for('submit_page')}}">
    <input class="form-control" name = 'search' type="search" placeholder="Search" aria-label="Search">
    <button class="btn submit-btn" type="submit">Search</button>
</form>
```

其對應的function為
```python=
@app.route('/submit_page', methods=['POST'])
def submit_page():
    search = request.values['search']
```
由上面可以知道search就是從html傳遞過來的變數。

#### 檔案上傳
首先需要一個enctype屬性設定為`multipart/form-data`的HTML表單，將該文提交到指定URL。 URL處理程式從`request.files[]`物件中提取檔案並將其儲存到所需的位置。可以從request.files [file]物件的filename屬性中獲取。 但建議使用secure_filename()函式獲取它的安全版本。

```python=
IMG_FOLDER = os.path.join('static', 'img') 
app.config['IMG_FOLDER'] = IMG_FOLDER #取得檔案路徑
os.path.join(app.config['IMG_FOLDER'], 'image.png')
```
#### Markup方法
[Flask 官方文件](https://dormousehole.readthedocs.io/en/latest/api.html#flask.Markup)
> **class flask.Markup(base='', encoding=None, errors='strict')**
> 
> Def : A string that is ready to be safely inserted into an HTML or XML document, either because it was escaped or because it was marked safe.
>Passing an object to the constructor converts it to text and wraps it to mark it safe without escaping. To escape the text, use the escape() class method instead.

這次專案中我發現這是一個很好用的方法，當你在後台嵌入HTML格式，經過route傳遞到.HTML會被自動轉譯成string格式而不是保留html（例如`<class=...>`會被轉型成 `&lt;class=...&gt;`），此時就需要使用到Markup。

##### 安裝Markup
`pip3 install Markup`
##### import
`from markupsafe import Markup`
##### 範例
```python=
Markup("<em>Hello</em> ") + "<foo>"
# Markup('<em>Hello</em> &lt;foo&gt;')
```
可以看到有Markup的被保留下來．

## Bootstrap 5
> [官方文件](https://getbootstrap.com/docs/5.0/getting-started/introduction/)
> [中文版](https://bootstrap5.hexschool.com)

這次的互動樣式都是使用 **Bootstrap5**建構出來的，他厲害的地方在於Bootstrap將流變隔線規劃的非常完整，要做RWD（響應式設計）會輕鬆許多。
此次專案主要用到幾項小元件
### Narbar
他是一個導覽列，我主要的功能按鈕都放在這個導覽列上
![](https://i.imgur.com/cAhECAA.png)
`Choose File: `選擇Xml(download from pubmed)或是Json(download from Twitter)
`Picture: `找出各篇的Zipf distribution、Porter's algo、stop word.
`Search: `搜尋相近字並highlight

### Navs and tabs
頁籤功能，主要是拿來做相似字排列，當點到button時便會對應到存在該單字的文章。
![](https://i.imgur.com/gHpYLni.png)

除此之外還有使用到了`Button`、`Alert`、`Dropdowns`等小功能。
另外還可以使用css更改顏色樣式，在使用Bootstrap前記得link css與javaScript
```htmlembedded=
<!- css ->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-wEmeIV1mKuiNpC+IOBjI7aAzPcEZeedi5yW5f2yOq55WWLwNGmvvx4Um1vskeMj0" crossorigin="anonymous">
```
```htmlembedded=
<!- javaScript ->
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js" integrity="sha384-IQsoLXl5PILFhosVNubq5LC7Qb9DXgDA9i+tQ8Zj3iwWAwPtgFTxbJ8NT4GN1R8p" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/js/bootstrap.min.js" integrity="sha384-lpyLfhYuitXl2zRZ5Bn2fqnhNAKOAaM/0Kr9laMspuaMiZfGmfwRNFh8HlMy49eQ" crossorigin="anonymous"></script>
```


## Text preprocessing

### What is Tokenization?
**Tokenization is the process by which a large quantity of text is divided into smaller parts called tokens.** These tokens are very useful for finding patterns and are considered as a base step for stemming and lemmatization. 

### 切割words
#### Python NLTK tokenize.WordPunctTokenizer()
用法：能從字母或是非字母字符流中提取

```python=
# import WordPunctTokenizer() method from nltk 
from nltk.tokenize import WordPunctTokenizer 
     
# Create a reference variable for Class WordPunctTokenizer 
tk = WordPunctTokenizer() 
     
# Create a string input 
gfg = "The price\t of burger \nin BurgerKing is Rs.36.\n"
     
# Use tokenize method 
geek = tk.tokenize(gfg) 
     
print(geek)

## output=
Output = [‘The’, ‘price’, ‘of’, ‘burger’, ‘in’, ‘BurgerKing’, ‘is’, ‘Rs’, ‘.’, ’36’, ‘.’]
```
#### Python nltk.tokenize.word_tokenize
```python=
from nltk.tokenize import word_tokenize
text = "God is Great! I won a lottery."
print(word_tokenize(text))

## output=
Output: ['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']
```
#### Python re.split

這也是我此次專案使用的方法
```python=
re.split(r'[!(\')#$"&…%^*+,./{}[;:<=>?@~ \　]+', word)
```

### 切割sentences

#### Python nltk.tokenize.PunktWordTokenizer
```python=
from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()
tokenizer.tokenize("Can't is a contraction.")

## output=
Output: ["Can't is a contraction."]
tokenizer.tokenize("Can't is a contraction. So is hadn't.")

## output=
Output: ["Can't is a contraction.", "So is hadn't."]
```
#### Python nltk.tokenize.sent_tokenize
```python=

from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery."
print(sent_tokenize(text))

Output: ['God is Great!', 'I won a lottery ']
```

### 詞性標註
#### Python nltk.corpus.treebank

文字和標注的組合是以tuple的方式儲存的
```python=
import nltk
from nltk.corpus import treebank

nltk.download('treebank')
print(treebank.tagged_sents()[0])

Output: [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
```

完整的標籤列表[參考](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

## Algorithm

### zipf distribution
![](https://i.imgur.com/zKgYVKR.png)
分母為Riemann Zeta function.

#### scipy.special.zetac(x)
此函數定義為
![](https://i.imgur.com/BB9QvYb.png)

> 此次我定義X軸為單字(從左至右frequency由高到低)
> Y軸是出現次數
> 總共取前50個單字來plot

### Porter's algo
Porter's algo 主要是將一些動詞類的變化去做統一的處理
例如`testing` `tested` `tests` 在經過Porter處理後，會統一變成`test`，對於字元的搜索會變得更加方便準確。

### Stop words
Stop words主要是表示一些定冠詞、介系詞、代名詞類的單字，因為在文章中出現的頻率會遠大於其他的單字，且並沒有什麼資訊，若是將這些單字留下來的話因為頻率過大的關係，有可能會擾亂到重要的字的分析。

### 相似字搜尋
[6.3. difflib — Helpers for computing deltas](https://docs.python.org/3.6/library/difflib.html#sequencematcher-examples)
![](https://i.imgur.com/Rqe1Z4X.png)

這個方法是會將最相近的word都找出來，`cutoff`是相似度，越高則挑選的越嚴格；`n`是會找出的字。


## Demo
`./`
首頁
![](https://i.imgur.com/xlCM5jM.png)


`./show_text`
選擇.xml或是.json搜尋所有的文本並顯示出來

![](https://i.imgur.com/FfGW8kd.png)

`./search`
會找到所有的相似字，並highlight起來

![](https://i.imgur.com/MMZglg5.png)
`./show_img`
顯示
* zipf Distribution
* Porter's algo
* using stop word 

![](https://i.imgur.com/ltUAVjN.png)
## Reference
https://ithelp.ithome.com.tw/articles/10197223

[Bootstrap 5](https://getbootstrap.com/docs/5.0/getting-started/introduction/)

[Python網頁設計：Flask使用筆記(二)- 搭配HTML和CSS](https://yanwei-liu.medium.com/python網頁設計-flask使用筆記-二-89549f4986de)

[使用 NLTK 搭配 Twitter API 拿取社群資料：以川普的 Twitter資料為例](https://cyeninesky3.medium.com/使用-nltk-搭配-twitter-api-拿取社群資料-以川普的-twitter資料為例-2bd493f452a6)

[iOS Twitter API串接 + JSON解析 - Twrendings](https://medium.com/彼得潘的-swift-ios-app-開發教室/ios-twitter-api串接-json解析-twrendings-4e0da5599398)

[flask許多神奇功能](https://hackmd.io/@shaoeChen/HJiZtEngG/https%3A%2F%2Fhackmd.io%2Fs%2FrkgXYoBeG)

[Flask example with POST](https://stackoverflow.com/questions/22947905/flask-example-with-post
)

[Flask example with loop](https://stackoverflow.com/questions/20317456/looping-over-a-tuple-in-jinja2)

[Python深度學習筆記(五)：使用NLTK進行自然語言處理](https://yanwei-liu.medium.com/python深度學習筆記-五-使用nltk進行自然語言處理-24fba36f3896)

[Stemming and Lemmatization in Python](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

[詞幹提取](https://zh.wikipedia.org/wiki/词干提取)

[Zipf's Law in NLP](https://iq.opengenus.org/zipfs-law/)

[Another Twitter sentiment analysis with Python — Part 3 (Zipf’s Law, data visualisation)](https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-3-zipfs-law-data-visualisation-fc9eadda71e7)



