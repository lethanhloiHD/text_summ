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

dim = 300

class W2V(object):
    def __init__(self):
        pass

    def pre_process(self, data):
        "tokenize text and remove all token with len = 1"
        result = []
        for row in data:
            row = pre_process_text(row, remove_number_punctuation=True)
            tokens = tokenizer.tokenize(row)
            tokens_ = []
            for token in tokens:
                if len(token) > 1:  # check word have more than 1 character
                    tokens_.append(token)
            result.append(tokens_)
        return result

    def build_model_w2v(self, data_, train_continue = False):
        if not train_continue:
            data = self.pre_process(data_)
            models = Word2Vec(data, size=dim, window=5, workers=4, min_count=2, iter=10, sg=1)
            with open(model_w2v_file, 'wb') as f:
                pickle.dump(models, f)

        elif train_continue:
            with open(model_w2v_file, 'rb') as f:
                models = pickle.load(f)
            data = self.pre_process(data_)
            models.train(data)
            with open(model_w2v_file, 'wb') as f:
                pickle.dump(models, f)

    def load_model(self, ):
        with open(model_w2v_file, 'rb') as f:
            model_w2v = pickle.load(f)
        return model_w2v

    def avg_representation_w2v_tfidf(self, tf, models, feature_name, model_w2v, sentence , remove_sw = False):
        if remove_sw :
            sentence = remove_stopwords(sentence)
        sentence = pre_process_text(sentence, remove_number_punctuation=True)
        words = set(tokenizer.tokenize(sentence.strip().lower()))
        tfidf_score = tf.get_tfidf_word_in_sentence(models, feature_name, sentence)
        words_keys = tfidf_score.keys()

        sentence_pre = np.zeros(dim)
        number_word = 0

        for word in words:
            if  (word in model_w2v.wv.vocab):
                if (word in words_keys) :
                    w_vec = np.array(model_w2v[word])
                    tf_vec = tfidf_score[word]
                    word_pre = w_vec * tf_vec
                    # word_pre = w_vec
                    sentence_pre += word_pre
                elif word not in words_keys :
                    w_vec = np.array(model_w2v[word])
                    sentence_pre += w_vec
                number_word +=1

        sentence_pre = (sentence_pre / max(number_word, 1))
        return sentence_pre

    def get_cosine_similary_w2v_tfidf(self, tf, models, feature_name, model_w2v, sentence1, sentence2 , remove_sw = False):

        pre1 = self.avg_representation_w2v_tfidf(tf, models, feature_name, model_w2v, sentence1, remove_sw)
        pre2 = self.avg_representation_w2v_tfidf(tf, models, feature_name, model_w2v, sentence2, remove_sw)
        score_cosine = cosine_similarity_vector(pre1, pre2)

        return score_cosine
