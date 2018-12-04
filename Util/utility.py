import csv, os
import math
import numpy as np
# from numpy.linalg import norm
import re
import nltk
from config import *


def remove_punctuation(pattern, phrase):
    for pat in pattern:
        return (re.findall(pat, phrase))


def pre_process_data(text, remove_number_punctuation = False):
    """ normalize string """
    if remove_number_punctuation :
        cont = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
        cont = "".join(remove_punctuation(pattern, cont))
    else :
        cont = text
    for i, v in mapping.items():
        if i in cont:
            cont = cont.replace(i, v)
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
    """compute cosine similarity of v1 to v2: (dot (v1,v2)/{||v1||*||v2||)"""
    dot = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot / (norm_vector1 * norm_vector2)


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

    for file in dir[:1]:
        print(file)
        _file_csv = os.path.join(directory_train, file)
        with open(_file_csv, encoding='utf-8') as file_:
            reader = csv.DictReader(file_)
            for row in reader:
                title = row['title_token'].strip().lower()
                sapo = row['sapo_token'].strip().lower()
                content = row['content_token'].strip().lower()
                cont = (title + " " + sapo + " " + content )
                cont = pre_process_data(cont)
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
        lenght_row = 0
        for row in reader:
            cont = row['Full text']
            summ = row['Summary (< 150 syllable)']
            content.append({
                "text": cont,
                "summ":summ
            })
            lenght_row +=1
    return content,lenght_row


def load_data_train_skipthought(directory_train):
    result = []
    data = load_data_train(directory_train)
    with open("data/data_skipthought/dummy_corpus.txt", "w") as file:
        for row in data[:1] :
            sentences = split_sentences(row)
            for sen in sentences :
                s = pre_process_data(sen,remove_number_punctuation=True)

                if len(s) > 10 :
                    s = s.strip()+" ."+"\n"
                    s= s.replace(" . ."," .")
                    result.append(s)
                    print(s)
                    file.write(s)

    print("number sentences :",len(result))
    return result




