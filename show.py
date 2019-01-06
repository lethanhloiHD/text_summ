from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
import os
import time
from src.modules.lda import *
import matplotlib.pyplot as plt
import numpy as np


x = [10,20,30,40,50]
tfidf = [0.2286,0.2521,0.2458,0.2438,0.2315]
word2vec =[0.3066,0.2544,0.2496,0.2573,0.2162]
doc2vec= [0.2599,0.2436,0.2297,0.2341,0.2474]
autoencoder = [0.2180,0.2405,0.2803,0.2712,0.2446]
plt.plot(x, tfidf,label='tfidf')
plt.plot(x, word2vec, label='word2vec')
plt.plot(x, doc2vec, label='doc2vec')
plt.plot(x, autoencoder, label='autoencoder')

plt.xlabel('Ratio number of sentences')
plt.ylabel('ROUGE2-P')

plt.title("Text Summarization")

plt.legend()

plt.show()

