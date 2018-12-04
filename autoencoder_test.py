import re
from config import *
from nltk.tokenize import RegexpTokenizer
import nltk
from Util.utility import *
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model,load_model
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from scipy.stats import describe
import collections
import nltk
import numpy as np
import os

VOCAB_SIZE  = 5000
SEQUENCE_LEN = 50
LATENT_SIZE = 512
NUM_EPOCHS=5
EMBED_SIZE = 200
BATCH_SIZE = 2

with open(model_w2v_file, 'rb') as f:
    model_w2v = pickle.load(f)

x_train = load_data_train_skipthought(directory_train)

# x_train = [
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies',
#     'viet nam vo dich',
#     'viet nam vo dich giai dau AFF 2018',
#     'Ha Noi tro ret trong thang 1',
#     'Mua ve tran dau Viet Nam  vs Philipies'
# ]
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
# length = []
# for x in x_train:
#     length.append(len(x.split()))
# max_lenght = max(length)

x_train_seq = pad_sequences(sequences, maxlen=SEQUENCE_LEN)

embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_SIZE))

for word, i in tokenizer.word_index.items():
    if word in model_w2v:
        embedding_vector = model_w2v[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

def sentence_generator(X, embeddings, batch_size):
    while True:
        print("loop once per epoch")
        num_recs  = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            Xbatch = embeddings[X[sids, :]]
            yield Xbatch, Xbatch
            print(Xbatch.shape)

train_size = 0.7
Xtrain, Xtest = train_test_split(x_train_seq, train_size=train_size)
train_gen = sentence_generator(Xtrain, embedding_matrix, BATCH_SIZE)
test_gen = sentence_generator(Xtest, embedding_matrix, BATCH_SIZE)

#####################################################################
string = ['viet nam vo dich AFF CUP 2018']
len_string  = len(string[0].split())
print(len_string)
tokenizer_string = Tokenizer(num_words=100)
tokenizer_string.fit_on_texts(string)
sequences_string = tokenizer.texts_to_sequences(string)
string_seq = pad_sequences(sequences_string, maxlen=SEQUENCE_LEN)
string_embedding_matrix = np.zeros((50, EMBED_SIZE))
for word, i in tokenizer_string.word_index.items():
    if word in model_w2v:
        embedding_vector = model_w2v[word]
        if embedding_vector is not None:
            string_embedding_matrix[i] = embedding_vector
string_gen = sentence_generator(string_seq,string_embedding_matrix,1)


########################################################################
inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum",
    name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True),
    merge_mode="sum",
    name="decoder_lstm")(decoded)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer="sgd", loss="mse")

num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
checkpoint = ModelCheckpoint(filepath="autoencoder.h5", save_best_only=True)
autoencoder.fit_generator(train_gen,
    steps_per_epoch=num_train_steps,
    epochs=NUM_EPOCHS,
    validation_data=test_gen,
    validation_steps=num_test_steps,
    callbacks=[checkpoint,early_stopping])

encoder = Model(autoencoder.input, autoencoder.get_layer("encoder_lstm").output)
# encoder = load_model("autoencoder.h5")
# xtest, ytest = string_gen.__next__()
# a  = encoder.predict(ytest)
# print(a[0].shape, a[0])
def compute_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))
#
k = 5
cosims = np.zeros((k))
i = 0

for bid in range(num_test_steps):
    xtest, ytest = test_gen.__next__()
    print("xtest :",xtest.shape)
    print("ytest :",ytest.shape)
    ytest_ = autoencoder.predict(xtest)
    print("ytest : ",ytest_.shape)
    Xvec = encoder.predict(ytest)
    print("Xvec :",Xvec.shape)
    Yvec = encoder.predict(ytest_)
    print("Yvec :",Yvec.shape)
    for rid in range(Xvec.shape[0]):
        if i >= k:
            break
        cosims[i] = compute_cosine_similarity(Xvec[rid], Yvec[rid])
        if i <= 10:
            print("cosine ",cosims[i])
            i += 1
    if i >= k:
        break