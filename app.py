from logging import debug
from flask import Flask, render_template, request, redirect, send_from_directory, Response
import os
import xml.etree.ElementTree as ET
import re
import json
from operator import itemgetter 
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import difflib
import pickle
import torch
from model import CBOW_Model, SkipGram_Model
from scipy.special import softmax
from markupsafe import Markup
from sklearn.decomposition import PCA
import itertools


IMG_FOLDER = os.path.join('static', 'img')

app = Flask(__name__)
app.config['IMG_FOLDER'] = IMG_FOLDER
TEXT = []
app.config['TEXT'] = TEXT
ROUTE = ''
app.config['ROUTE'] = ROUTE

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/show_text')
def show_text():
    app.config['ROUTE'] = request.url_rule
    print('text', app.config['ROUTE'] )
    _, doc = find_XML()
    
    return render_template('show_text.html', doc = doc)

@app.route('/show_json')
def show_json():
    app.config['ROUTE'] = request.url_rule
    _, doc = find_JSON()

    return render_template('show_json.html', doc = doc)

@app.route('/show_img')
def show_img():
    print(app.config['ROUTE'])
    path = app.config['ROUTE']
    if '/show_text' == str(path):
        xml4000_Figure = os.path.join(app.config['IMG_FOLDER'], 'test4000_Figure_2.png')
        xml4000_Porter = os.path.join(app.config['IMG_FOLDER'], 'test4000_porter_2.png')
        xml1000_Figure = os.path.join(app.config['IMG_FOLDER'], 'test1000_Figure_2.png')
        xml1000_Porter = os.path.join(app.config['IMG_FOLDER'], 'test1000_porter_2.png')
        xml2000_Figure = os.path.join(app.config['IMG_FOLDER'], 'test2000_Figure_2.png')
        xml2000_Porter = os.path.join(app.config['IMG_FOLDER'], 'test2000_porter_2.png')
        xml4000_stop_Figure = os.path.join(app.config['IMG_FOLDER'], 'test4000_stop_Figure_2.png')
        xml4000_stop_Porter = os.path.join(app.config['IMG_FOLDER'], 'test4000_stop_Porter_2.png')
        xml1000_stop_Figure = os.path.join(app.config['IMG_FOLDER'], 'test1000_stop_Figure_2.png')
        xml1000_stop_Porter = os.path.join(app.config['IMG_FOLDER'], 'test1000_stop_Porter_2.png')
        xml2000_stop_Figure = os.path.join(app.config['IMG_FOLDER'], 'test2000_stop_Figure_2.png')
        xml2000_stop_Porter = os.path.join(app.config['IMG_FOLDER'], 'test2000_stop_Porter_2.png')
    else:
        json_Figure = os.path.join(app.config['IMG_FOLDER'], 'json1000_Figure.png')
        json_Porter = os.path.join(app.config['IMG_FOLDER'], 'json1000_porter.png')
        json_stop_Figure = os.path.join(app.config['IMG_FOLDER'], 'json1000_stopWord_Fig.png')
        json_stop_Porter = os.path.join(app.config['IMG_FOLDER'], 'json1000_stopWord_porter.png')
    
    return render_template("show_img.html", **locals())

@app.route('/submit_page', methods=['POST'])
def submit_page():
    path = app.config['ROUTE']
    print(path)
    search = request.values['search']
    print(search)
    
    if '/show_text' == str(path):
        word_dict = {}
        words, doc = find_XML()
        edit_dist_word = difflib.get_close_matches(search.lower(), words, n = len(words), cutoff = 0.7)
        word_set = set(edit_dist_word)
        for word in word_set:
            if (edit_dist_word.count(word) > 3 and len(word) > 2):
                word_dict[word] = edit_dist_word.count(word)
        
        edit_dist_word = sorted(word_dict.items(), key = lambda x: x[1], reverse=True)
        print(word_dict)
        print(edit_dist_word)
        new_list = []
        
        match_article = {}
        for key_word in edit_dist_word:  
            print(key_word)
            doc_list = []
            new_doc = {}
            for title, article in doc.items():
                if title and article:
                    new_doc = {}
                    find_pattern = re.compile(key_word[0], re.I)
                    '''Match Title'''
                    match_title = find_pattern.findall(title)
                    new_text_doc = {}
                    if match_title:
                        '''吻合'''
                        for word in match_title:
                            replaced_word = "<font style =\'background:#B1BEB9;\'>" + word + "</font>"
                            new_title = title.replace(word, replaced_word)  #更新後的title
                            for label, inner_text in article.items():   
                                if inner_text:
                                    match_text = find_pattern.findall(inner_text)
                                    if match_text:
                                        '''吻合'''
                                        for inner_word in match_text:
                                            replaced_innerword = "<font style =\'background:#B1BEB9;\'>" + inner_word + "</font>"
                                            new_text = inner_text.replace(inner_word, replaced_innerword)
                                        new_text_doc[label] = Markup(new_text)
                                    else:
                                        new_text_doc[label] = inner_text
                                else:
                                    new_text_doc[label] = 'No Article.'
                                # print(new_text_doc)
                            new_doc[Markup(new_title)] = new_text_doc
                        doc_list.append(new_doc)
                    else: #沒有匹配到title 看內文
                        for label, inner_text in article.items():   
                            if inner_text:
                                match_text = find_pattern.findall(inner_text)
                                if match_text:
                                    '''吻合'''
                                    for inner_word in match_text:
                                        replaced_innerword = "<font style =\'background:#B1BEB9;\'>" + inner_word + "</font>"
                                        new_text = inner_text.replace(inner_word, replaced_innerword)
                                    new_text_doc[label] = Markup(new_text)
                                    new_doc[title] = new_text_doc
                                    doc_list.append(new_doc)
            match_article[key_word] = doc_list
        new_list.append(match_article) 
                    
    else:
        word_dict = {}
        words, doc = find_JSON()
        
        edit_dist_word = difflib.get_close_matches(search.lower(), words, n = len(words), cutoff = 0.6)
        word_set = set(edit_dist_word)
        for word in word_set:
            word_dict[word] = edit_dist_word.count(word)
        
        edit_dist_word = sorted(word_dict.items(), key = lambda x: x[1], reverse=True)
        print(word_dict)
        print(edit_dist_word)
        new_list = []

        match_article = {}
        for key_word in edit_dist_word: 
            find_pattern = re.compile(key_word[0], re.I)
            doc_list = []
            for name, article in doc.items():
                if name and article:
                    new_doc = {}  
                    match_text = find_pattern.findall(article)
                    if match_text:
                        for word in match_text:
                            replaced_word = "<font style =\'background:#fe7654;\'>" + word + "</font>"
                            new_text = article.replace(word, replaced_word)    #更新後的aritcle
                        match_title = find_pattern.findall(name)
                        if match_title:
                            for word in match_title:
                                replaced_word = "<font style =\'background:#fe7654;\'>" + word + "</font>"
                                new_name = name.replace(word, replaced_word)
                            new_doc[Markup(new_name)] = Markup(new_text)
                        else:
                            new_doc[name] = Markup(new_text)
                        doc_list.append(new_doc)
                    else: #沒有匹配到內文 單純找title
                        match_title = find_pattern.findall(name)
                        if match_title:
                            for word in match_title:
                                replaced_word = "<font style =\'background:#fe7654;\'>" + word + "</font>"
                                new_name = name.replace(word, replaced_word)
                            new_doc[Markup(new_name)] = article

                            doc_list.append(new_doc)
            if doc_list:
                match_article[key_word] = doc_list
        new_list.append(match_article)
    return render_template("submit_page.html", **locals())

@app.route('/skip_gram')
def skip_gram():
    sg_search = "Input Word"
    return render_template("SkipGram.html")

@app.route('/show_skipGram', methods=['POST'])
def show_skipGram():
    sg_search = request.values['sg_search']
    print(sg_search)
    with open('./file/word2idx_lemma.pickle','rb') as file:
        word2idx = pickle.load(file)
    with open('./file/idx2word_lemma.pickle','rb') as file:
        idx2word = pickle.load(file)
    predict_word = {}
    if sg_search.lower() not in word2idx.keys():
        word = sg_search
    else:
        folder = "Model"
        model = SkipGram_Model(vocab_size = len(word2idx), embedding_dim = 600)
        device = torch.device("cpu")
        # model = model.to(device)
        ## Load model ...
        model.load_state_dict(torch.load(f"./{folder}/SkipGram_lemma_65.pt", map_location=device))
        model.eval()
        test_voc = word2idx[sg_search.lower()]
        inputs = torch.tensor(test_voc)
        with torch.no_grad():
            inputs = inputs.to(device)
            test_pred = model(inputs)
            test_label = softmax(test_pred.squeeze().cpu().data.numpy())
            idx = np.sort(np.argpartition(test_label, -15)[-15:])
            print(idx)
            display_plot(model, idx, idx2word)
            for i in idx[::-1]:
                predict_word[idx2word[i]] = test_label[i]
                print(f'{idx2word[i]} : {test_label[i]}')
        predict_word = {k: v for k, v in sorted(predict_word.items(), key=lambda item: item[1], reverse=True)}
        pred_Figure = os.path.join(app.config['IMG_FOLDER'], 'predict_pic.png')

    return render_template("show_skipGram.html", **locals())

def display_plot(model, idx, idx2word):
    # Take word vectors
    word_vectors = []
    for word in idx:
        inputs = torch.tensor(word)
        pred = model(inputs)
        pred = softmax(pred.squeeze().cpu().data.numpy())
        word_vectors.append(pred)
        
    two_dim = PCA().fit_transform(np.array(word_vectors))[:,:2]
    # Draw
    plt.switch_backend('agg')
    plt.figure(figsize=(10,10))
    color_cycle= itertools.cycle(["orange","pink","blue","brown","red","grey","yellow","green", "#ECD5FA", "#E8C88C"])
    for word, (x, y) in zip(idx, two_dim):
        plt.scatter(x, y, label= idx2word[word], color=next(color_cycle))
        plt.text(x+0.001, y+0.001, idx2word[word])
    plt.legend(loc='best')
    plt.savefig('./static/img/predict_pic.png')


def find_XML():
    with open('./file/xml4000_words.pickle', 'rb') as file:
        words = pickle.load(file)
    with open('./file/xml4000_doc.pickle', 'rb') as file:
        doc = pickle.load(file)
    return words, doc

def find_JSON():
    with open('./file/json1000_words.pickle', 'rb') as file:
        words = pickle.load(file)
    with open('./file/json1000_doc.pickle', 'rb') as file:
        doc = pickle.load(file)
    return words, doc



'''字串變成單字'''
def text2word(text): 
    '''v1'''
    # words = re.split(r'[!()#$%^*+,./;:<=>?@~]+', text)
    # # nltk.download('punkt')
    # words = ' '.join(str(x) for x in words)
    # words = word_tokenize(words)
    '''v2'''
    words = []
    # split_word = text.split(']')
    # text = ' '.join(str(x) for x in split_word)
    split_word = text.split("'s")
    for word in split_word:
        words.append(re.split(r'[\'\]!()#$"&%^*+,./{}[;:<=>?@~ ]+', word))
    split_words = [i for item in words for i in item]


    return split_words
    

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port='8080')