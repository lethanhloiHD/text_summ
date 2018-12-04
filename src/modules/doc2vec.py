from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from Util.utility import *
from config import  *
from nltk.tokenize import RegexpTokenizer
import pickle

tokenizer = RegexpTokenizer(r'\w+')


class D2V(object):
    def __init__(self):
        pass

    def pre_process(self,data):
        result = []
        for row in data :
            sentences = split_sentences(row)
            for sent in sentences :
                sent_ = sent.replace("_"," ")
                tokens = tokenizer.tokenize(sent_)
                if len(tokens) > 4 :
                    sent = pre_process_data(sent,remove_number_punctuation=True)
                    result.append(sent)
        return result

    def build_model(self, data,save = True):
        data_sentences = self.pre_process(data)
        print("number senteces :", len(data_sentences))
        tagged_data = [TaggedDocument(words=word_tokenize(d.lower()),
                                      tags=[str(i)]) for i, d in enumerate(data_sentences)]
        max_epochs = 10
        alpha = 0.025
        model = Doc2Vec(vec_size=200,alpha=alpha,min_alpha=0.0025,min_count=2,dm=1)
        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        if save :
            with open(model_d2v_file, 'wb') as f:
                pickle.dump(model, f)

    def load_model(self):
        with open(model_d2v_file, 'rb') as f:
            model = pickle.load(f)
        return model

    def get_vector_sentences(self,model, sentence ):
        token_split = tokenizer.tokenize(sentence)
        tokens= []
        for t in token_split :
            if len(t) > 1 :
                tokens.append(t)
        # tokens = token_split
        vector = model.infer_vector(tokens)
        return vector

    def get_cosine_similary(self, model,sentence1, sentence2):
        vector1 = self.get_vector_sentences(model,sentence1)
        vector2 = self.get_vector_sentences(model,sentence2)
        score = cosine_similarity_vector(vector1,vector2)

        return score

