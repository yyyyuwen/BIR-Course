import sys
import os
import xml.etree.ElementTree as ET
import re
import json
from operator import itemgetter 
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
from nltk.corpus import stopwords, wordnet

import nltk


'''讀取資料'''
def read_xmlfile():
    tree = ET.parse('./file/test4000.xml')
    root = tree.getroot()
    Article = root.findall("PubmedArticle")
    text_list = []
    doc = {}
    for elem in Article:
        article_text = {}
        title = elem.find("MedlineCitation").find("Article").find("ArticleTitle").text
        text_list.append(elem.find("MedlineCitation").find("Article").find("ArticleTitle").text) ##標題
        for article in elem.find("MedlineCitation").find("Article").findall("Abstract"): #內文位置 
            
            if (article.find('AbstractText') is not None):
                abs = article.find('AbstractText')
                if 'Label' in abs.attrib:
                    article_text[abs.attrib['Label']] =  abs.text
                    text_list.append(abs.attrib['Label'])
                else:
                    article_text['Abstract'] =  abs.text
                text_list.append(abs.text)
            else:
                article_text['Abstract'] =  'no article.'
        doc[title] = article_text
    text = ' '.join(str(x) for x in text_list)
    text = text.replace('\n', ' ')
    return text, doc

def read_jsonfile():
    with open('./file/data0_1000.json', 'rb') as json_file:
        json_data = json.load(json_file)
        text_list = []
        doc = {}
        for data in json_data:
            fullname = data.get('name')
            username = data.get('screen_name')
            text = data.get('text')
            name = fullname + '@' + username
            text_list.append(name)
            text_list.append(text)
            doc[name] = text
        text = ' '.join(str(x).lower() for x in text_list)
        text = text.replace('\n', ' ')
    return text, doc

'''字串變成單字'''
def text2word(text): 
    words = []
    split_word = text.split(']')
    text = ' '.join(str(x).lower() for x in split_word)
    split_word = text.split("'s")
        
    for word in split_word:
        words.append(re.split(r'[!(\')#$"&…%^*+,./{}[;:<=>?@~ \　]+', word))
    split_words = [i for item in words for i in item]
    # words = re.split(r'[ ]', words)
    # split on white-space 
    # word_list = re.split(r"\s+", target_string) 
    # nltk.download('punkt') 
    # print(words)
    # words = ' '.join(str(x) for x in words)
    # words = word_tokenize(words)

    return split_words

'''字串變成句子'''
def text2sent(text):
    sentences = sent_tokenize(text)
    sentences = [re.sub(r'[^a-z0-9|^-]', ' ', sent.lower()) for sent in sentences]
    clean_words = []
    for sent in sentences:
        words = [word for word in sent.split() if not word.replace('-', '').isnumeric()]
        words = stop_word(words)
        clean_words.append(' '.join(words))
    return clean_words

'''將sentences 切成 words'''
def splitsent2words(text):
    tokens = [x.split() for x in text]
    return tokens

'''字&標籤互相對應'''
def word_idx(text):
    vocabulary = []
    for word in text:
        if word not in vocabulary:
            vocabulary.append(word)
    
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    return word2idx, idx2word

'''get pair'''
def get_pair(split_words, word_idx):
    window_size = 2
    idx_pairs = []
    # for each sentence
    indices = [word_idx[word] for word in split_words]
    # for each word, threated as center word
    for sent_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            word_pos = sent_pos + w
            if word_pos < 0 or word_pos >= len(indices) or sent_pos == word_pos:
                continue
            sent_idx = indices[word_pos]
            idx_pairs.append((indices[sent_pos], sent_idx))
    print(idx_pairs[:20])
    return np.array(idx_pairs)

'''zipf distribution'''
def zipf(text):
    frequency = {}
    for word in text:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    # for key, value in reversed(sorted(frequency.items(), key = itemgetter(1))):
    #     print (key, value)
    frequency = {key: value for key, value in reversed(sorted(frequency.items(), key = lambda item: item[1]))}
    # print(frequency)
    
    #convert value of frequency to numpy array
    s = np.array(list(frequency.values()))[:50]
    
    '''Calculate zipf and plot the data'''
    a = 2. #  distribution parameter has to be >1
    plt.figure(figsize=(20,20))
    count, bins, ignored = plt.hist(s[s< 50], 50, density=True)
    plt.title("Zipf plot")
    x = np.arange(len(s))
    plt.xlabel("Frequency Rank of Token")
    # y = s[x]**(-a) / special.zetac(a)
    y = special.zetac(a) / s[x]**(-a)
    plt.ylabel("Absolute Frequency of Token")
    xlabels = list(frequency.keys())[:50]
    plt.xticks(x, xlabels, rotation=90)
    plt.plot(x, y/max(y), linewidth=2, color='r')
    plt.grid(True)
    plt.show()

    '''Calculate zipf for rank and plot the data'''
    plt.figure(figsize=(20,20))
    plt.title("Zipf plot using stop words")
    x = np.arange(len(s))
    plt.xlabel("Frequency Rank of Token")
    y = s[x]
    plt.ylabel("Frequency of words")
    xlabels = list(frequency.keys())[:50]
    plt.xticks(x, xlabels, rotation=90)
    plt.plot(x, y, linewidth=2, color='r')
    plt.show()


    '''bar plot'''
    # plt.figure(figsize=(20,20))  #to increase the plot resolution
    # plt.ylabel("Frequency")
    # plt.xlabel("Words")
    # word = list(frequency.keys())[:50]
    # freq = list(frequency.values())[:50]
    # print(freq)
    # x = np.arange(len(word))
    # plt.bar(x, freq)
    # plt.xticks(x, word, rotation=90)    #to rotate x-axis values
    # plt.show()

def porter(words):
    ps = PorterStemmer()
    porter_list = []
    for word in words:
        porter_list.append(ps.stem(word))

    return porter_list

def stop_word(words):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    return filtered_sentence

def lemma(sentences):
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
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
    return lemma_word




def main():
    text, doc = read_xmlfile()
    # text, doc = read_jsonfile()
    # words = text2word(text)
    # text = stop_word(text)
    text = text2sent(text)
    split_words = splitsent2words(text)
    lemma_words = lemma(split_words)
    print(lemma_words)
    word2idx, idx2word = word_idx(lemma_words)
    pair = get_pair(lemma_words, word2idx)
    # with open('./file/word2idx_lemma.pickle', 'wb') as f:
    #     pickle.dump(word2idx, f)
    # with open('./file/idx2word_lemma.pickle', 'wb') as f:
    #     pickle.dump(idx2word, f)
    # with open('./file/pairword_lemma.pickle', 'wb') as f:
    #     pickle.dump(pair, f)


    
    
    # zipf(words)
    # words = porter(words)
    # zipf(words)
    # with open('./xml4000_words.pickle','wb') as file:
    #     pickle.dump(words, file)
    # with open('./xml4000_doc.pickle','wb') as file:
    #     pickle.dump(doc, file)
    # with open('./json1000_words.pickle','wb') as file:
    #     pickle.dump(words, file)
    # with open('./json1000_doc.pickle','wb') as file:
    #     pickle.dump(doc, file)
    

if __name__ == '__main__':
    main()