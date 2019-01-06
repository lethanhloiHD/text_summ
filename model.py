from src.modules.autoencoder import *
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
import os
import time
from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
from src.modules.lda import *
import matplotlib.pyplot as plt



def build_model_lda():
    data = load_data_train(directory_train)
    lda = LDA()
    lda.build_model_lda(data)


def build_model_autoencoder():
    with open(model_w2v_file, 'rb') as f:
        model_w2v = pickle.load(f)

    train_gen,test_gen,num_train_steps,num_test_steps = pre_process_autoencoder(model_w2v)

    train_model_autoencoder(train_gen,test_gen,num_train_steps,num_test_steps)


def build_model_w2v():
    data = load_data_train(directory_train)
    w2v = W2V()
    w2v.build_model_w2v(data)


def build_model_d2v():
    data = load_data_train(directory_train)
    d2v = D2V()
    d2v.build_model(data)


def build_model_tfidf():
    data = load_data_train(directory_train)
    tf= Tfidf()
    tf.build_model_featureNames(data)