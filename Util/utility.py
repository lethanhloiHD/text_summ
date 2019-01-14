import csv, os
import math
import numpy as np
# from numpy.linalg import norm
import re,json
import nltk
from config import *
from nltk.tokenize import RegexpTokenizer
import requests
from nltk import ngrams
tokenizer = RegexpTokenizer(r'\w+')

def init_tokenizer():
    with open(config_token) as config_buffer:
        config = json.loads(config_buffer.read())
    return config['url']

url_token = init_tokenizer()

def word_grams(sequence, min=1, max=2):
    words = sequence.split()
    results = []
    for n in range(min, max + 1):
        for ngram in ngrams(words, n):
            results.append(' '.join(str(i) for i in ngram))
    return results

def remove_punctuation(pattern, phrase):
    " remove pattern in phrase "

    for pat in pattern:
        return (re.findall(pat, phrase))


def remove_stopwords(setence):
    """ remove stopwords in sentence """
    sent =  setence.lower().strip()
    stopwords = load_stopwords(stopword_path)
    tokens = tokenizer.tokenize(sent)
    tokens_filter_stopwords = [word for word in tokens if word not in stopwords]
    string = " ".join(word for word in tokens_filter_stopwords)
    return string


def pre_process_text(text, remove_number_punctuation = True):
    """ normalize string """
    text = text.strip().lower()
    print(text)
    if remove_number_punctuation :
        cont = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
        cont = "".join(remove_punctuation(pattern, cont))
    else :
        cont = text
    for i, v in mapping.items():
        if i in cont:
            cont = cont.replace(i, v)
    print(cont)
    return cont


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    # sentence_delimiters = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    # sentences = sentence_delimiters.split(text)
    sentences = nltk.sent_tokenize(text)
    return sentences


def cosine_similarity_vector(vector1,vector2):
    """compute cosine similarity of v1 to v2:
        (dot (v1,v2)/{||v1||*||v2||)"""
    dot = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1,2)
    norm_vector2 = np.linalg.norm(vector2,2)
    return dot / (norm_vector1 * norm_vector2)


def score_position_sentence(total_sentences, position ):

    if total_sentences > 1 :
        pos = float(position/total_sentences)
        if 0 < pos <= 0.1 : return 0.17
        elif pos <= 0.2 : return 0.23
        elif pos <= 0.3 : return 0.14
        elif pos <= 0.4 : return 0.08
        elif pos <= 0.5 : return 0.05
        elif pos <= 0.6 : return 0.04
        elif pos <= 0.7 : return 0.06
        elif pos <= 0.8 : return 0.04
        elif pos <= 0.9 : return 0.04
        elif pos <= 1.0: return 0.15
    else :
        return 0

def load_stopwords(stoppath):
    stop_words = []
    for x in open(stoppath, 'r').read().split('\n'):
        d = ''
        w = x.split(" ")
        if len(w) == 1:
            stop_words.append(w[0])
        else:
            for i in range(len(w) - 1):
                d += w[i] + "_"
            d += w[len(w) - 1]
            stop_words.append(d)
    return stop_words


def load_data_train(directory_train, segmentation = True) :
    result = []
    dir = os.listdir(directory_train)
    for file in dir:
        print(file)
        _file_csv = os.path.join(directory_train, file)
        with open(_file_csv, encoding='utf-8') as file_:
            reader = csv.DictReader(file_)
            for row in reader:
                title = row['title_token'].strip()
                sapo = row['sapo_token'].strip()
                content = row['content_token'].strip()
                cont = (title + " " + sapo + " " + content )
                cont = pre_process_text(cont)
                if segmentation:
                    result.append(cont)
                else:
                    cont = (cont).replace("_", " ")
                    result.append(cont)
    return result


def load_data_test():
    content = []
    with open(file_data_test, encoding='utf-8') as file_:
        reader = csv.DictReader(file_)
        for row in reader:
            cont = row['Title']+". "+row['Full text']
            summ = row['Summary (< 150 syllable)']
            content.append({
                "text": cont,
                "summ":summ
            })
    return content


def load_data_train_skipthought(directory_train):
    result = []
    data = load_data_train(directory_train)
    with open("data/data_skipthought/dummy_corpus.txt", "w") as file:
        for row in data[:10000]:
            sentences = split_sentences(row)
            for sen in sentences :
                s = pre_process_text(sen, remove_number_punctuation=True)
                if len(s) > 10 :
                    result.append(s)
                    print(s)
                    file.write(s)

    print("number sentences :",len(result))
    return result


