from gensim.models import Word2Vec
import csv
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from config import *
from Util.utility import *
import pickle
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')
from sklearn.metrics.pairwise import cosine_similarity


class W2V(object):
    def __init__(self):
        pass

    def pre_process(self,data):
        "tokenize text and remove all token with len = 1"
        result = []
        for row in data :
            tokens = tokenizer.tokenize(row)
            for token in tokens :
                if len(token) > 1 :
                    result.append(tokens)
        return result

    def build_model_w2v(self ,data_,train_online = False) :
        if not train_online:
            data = self.pre_process(data_)
            models = Word2Vec(data,size=200,window=5,workers=4,min_count=1,iter=5,sg=1)
            with open(model_w2v_file, 'wb') as f:
                pickle.dump(models, f)

        elif train_online:
            with open(model_w2v_file, 'rb') as f:
                models = pickle.load(f)
            data = self.pre_process(data_)
            models.train(data)
            with open(model_w2v_file, 'wb') as f:
                pickle.dump(models, f)

    def avg_representation_w2v_tfidf(self, tf, setence):

        # model_w2v = Word2Vec.load(model_w2v_file)

        with open(model_w2v_file, 'rb') as f:
            model_w2v = pickle.load(f)
        words = set(tokenizer.tokenize(setence.strip().lower()))
        tfidf_score = tf.get_tfidf_word_in_sentence(setence)
        words_keys = tfidf_score.keys()

        sentence_pre = np.zeros(100)
        number_word = len(words)

        for word in words:
            if (word in words_keys) and (word in model_w2v.wv.vocab):
                w_vec = np.array(model_w2v[word])
                tf_vec = tfidf_score[word]
                word_pre = w_vec * tf_vec
                sentence_pre += word_pre

        sentence_pre = (sentence_pre / max(number_word, 1))
        return sentence_pre

    def get_cosine_similary_w2v_tfidf(self, tf, sentence1, sentence2):
        pre1 = self.avg_representation_w2v_tfidf( tf, sentence1)
        pre2 = self.avg_representation_w2v_tfidf( tf, sentence2)
        score_cosine = cosine_similarity_vector(pre1, pre2)
        print(score_cosine)

        return score_cosine