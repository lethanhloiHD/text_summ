import csv
import time
# from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from config import *
from Util.utility import *


def pre_process(data):
    result = []
    for row in data :
        content = pre_process_text(row, remove_number_punctuation= False)
        result.append(content)
    return result

class Tfidf(object):
    def __init__(self,):
        self.stopwords = load_stopwords(stopword_path)

    def build_model_featureNames(self,data):
        model = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                                stop_words=self.stopwords, min_df = 3, max_df=0.7)
        data = pre_process(data)
        model.fit(data)
        feature_names = model.get_feature_names()

        with open(file_model_tfidf, 'wb') as f:
            pickle.dump(model, f)
        with open(file_feature_names, 'wb') as f:
            pickle.dump(feature_names, f)

    def load_model_featureNames(self,):

        with open(file_model_tfidf, 'rb') as f:
            models = pickle.load(f)
        with open(file_feature_names, 'rb') as f:
            feature_name = pickle.load(f)

        return models,feature_name

    def get_tfidf_sentence(self,models, sentence,remove_sw = False):
        if remove_sw :
            sentence = remove_stopwords(sentence)
        sentence = [sentence.strip().lower()]
        tfidf = models.transform(sentence)
        return tfidf

    def get_tfidf_word_in_sentence(self,models, feature_names,text):
        sentence = [text.strip().lower()]
        tfidf_matrix = models.transform(sentence)
        feature_index = tfidf_matrix.nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[0, x] for x in feature_index])

        result = {}
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            result.update({w:s})
            # print(w,s)
        return result


    def cosine_similarity_tfidf(self,models, sentence1, sentence2, remove_sw = False):

        tfidf_sentence1 = self.get_tfidf_sentence(models,sentence1, remove_sw)
        tfidf_sentence2 = self.get_tfidf_sentence(models,sentence2,remove_sw)
        cosine_score = cosine_similarity(tfidf_sentence1,tfidf_sentence2)

        return cosine_score
