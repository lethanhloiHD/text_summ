from src.LexRank import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
import os
import time
from src.modules.lda import *


# data = load_data_train(directory_train)

# corpus, dic = pre_process_lad(data)

lda = LDA()
# lda.build_model_lda(data)

# model_lda , dictionary_lda = lda.load_model_lda()

lda.get_vector_lda("Samsumg ket hop voi Nokia")