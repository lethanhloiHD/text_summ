from gensim.corpora.dictionary import Dictionary
from gensim import models
import pickle
from Util.utility import *


def plMMR_score_pre(document_vector,sentence_vector ) :
    score = 0
    for topic in NUM_TOPICS :
        if topic in document_vector.keys() and topic in sentence_vector.keys() :
            score += document_vector[topic] *sentence_vector[topic]
    return score


def plMMR_score_af(document_vector,sentence_vector_i , sentence_vector_j) :
    score = 0
    for topic in NUM_TOPICS :
        if topic in document_vector.keys() and topic in sentence_vector_i.keys() \
                                           and topic in sentence_vector_j.keys():
            score += document_vector[topic] *sentence_vector_i[topic]* sentence_vector_j[topic]
    return score


def plMMR_score(lda,document_vector , candidiate_sentences,summary_set_sentences, lamda = 0.7):

    score_m = 0
    sentence_select = candidiate_sentences[0]
    for sentence in candidiate_sentences :
        sentence_vector_i  = lda.get_vector_text_lda(sentence)
        score_pre = plMMR_score_pre(document_vector,sentence_vector_i)
        score_afs = []
        if len(summary_set_sentences) > 0 :
            for sent in summary_set_sentences :
                sentence_vector_j = lda.get_vector_text_lda(sent)
                score_af = plMMR_score_af(document_vector,sentence_vector_i,sentence_vector_j )
                score_afs.append(score_af)
        score_max  = max(score_afs)
        score = lamda * score_pre + (1-lamda)*score_max
        if score > score_m :
            sentence_select = sentence
    return sentence_select


def plMMR(document, candidate_pagerank , limit_sentences):
    lda = LDA()
    document_vector = lda.get_vector_text_lda(document)
    summay_set_sentences = candidate_pagerank[0]
    candidates = candidate_pagerank[1:]

    while len(summay_set_sentences) < limit_sentences :
        sentence_select = plMMR_score(lda,document_vector,candidates,summay_set_sentences)
        candidates.remove(sentence_select)
        summay_set_sentences.append(sentence_select)

    return summay_set_sentences

def pre_process_lad(data_train):
    stoplist = load_stopwords(stopword_path)
    text_data = []
    for document in data_train :
        docs =[word for word in  document.lower().split()
               if (word not in stoplist and len(word) > 3)]
        text_data.append(docs)

    dictionary = Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    return corpus, dictionary


class LDA(object):
    def __init__(self):
        try:
            self.model_lda , self.dictionary_lda = self.load_model_lda()
        except :
            print("Please building models before load !")

    def load_model_lda(self):
        with open(save_lda, 'rb') as f:
            model_lda = pickle.load(f)
        with open(save_dictonary_lda, 'rb') as f:
            dictionary_lda = pickle.load(f)
        return model_lda , dictionary_lda

    def build_model_lda(self, data_train):
        corpus, self.dictionary_lda = pre_process_lad(data_train)
        self.model_lda = models.ldamodel.LdaModel(corpus = corpus,
                                                  id2word=self.dictionary_lda,
                                                  num_topics=NUM_TOPICS,
                                                  alpha='auto',
                                                  per_word_topics=True,
                                                  )
        topics = self.model_lda.print_topics(num_words=4)
        for topic in topics[:5]:
            print(topic)
        with open(save_lda, 'wb') as f:
            pickle.dump(self.model_lda, f)
        with open(save_dictonary_lda, 'wb') as f:
            pickle.dump(self.dictionary_lda, f)
        return self.model_lda, self.dictionary_lda

    def get_vector_text_lda(self,text):
        text = [text.lower().strip()]
        bow = self.dictionary_lda.doc2bow(text)
        vector = self.model_lda.get_document_topics(bow)
        print("get_document_topics", vector)
        topic_score = {}
        for item in vector :
            topic_score.update({ item[0] : item[1]})

        return topic_score