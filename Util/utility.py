import csv, os
import math
import numpy as np
# from numpy.linalg import norm
import re
import nltk
from config import *

def pre_process_data(text, remove_number = True):
    """ normalize string """
    if remove_number :
        cont  = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
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
    sentence_delimiters = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_delimiters.split(text)
    # sentences = nltk.sent_tokenize(text)
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

    for file in dir:
        print(file)
        _file_csv = os.path.join(directory_train, file)
        with open(_file_csv, encoding='utf-8') as file_:
            reader = csv.DictReader(file_)
            for row in reader:
                title = row['title_token'].strip().lower()
                sapo = row['sapo_token'].strip().lower()
                content = row['content_token'].strip().lower()
                cont = (title + ". " + sapo + ". " + content )
                cont = pre_process_data(cont)
                if segmentation:
                    result.append(cont)
                else:
                    cont = (cont).replace("_", " ")
                    result.append(cont)
    return result



def load_data_test(file_txt):
    with open(file_txt, "r") as file:
        content = file.read()
    return content