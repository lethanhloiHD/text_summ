# import re
# from config import *
# from nltk.tokenize import RegexpTokenizer
# import nltk
# from Util.utility import *
# import pickle
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers import Input
# from keras.layers.core import RepeatVector
# from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import Bidirectional
# from keras.models import Model,load_model
# from keras.preprocessing import sequence
# from keras.layers.embeddings import Embedding
# from scipy.stats import describe
# import collections
# import nltk
# import numpy as np
# import os
#
# VOCAB_SIZE  = 50000
# SEQUENCE_LEN = 50
# LATENT_SIZE = 512
# NUM_EPOCHS= 100
# EMBED_SIZE = 300
# BATCH_SIZE = 64
#
# with open(model_w2v_file, 'rb') as f:
#     model_w2v = pickle.load(f)
#
# x_train = load_data_train_skipthought(directory_train)
#
# tokenizer = Tokenizer(num_words=VOCAB_SIZE)
# tokenizer.fit_on_texts(x_train)
# sequences = tokenizer.texts_to_sequences(x_train)
# length = []
# for x in x_train:
#     length.append(len(x.split()))
# max_lenght = max(length)
#
# x_train_seq = pad_sequences(sequences, maxlen=SEQUENCE_LEN)
#
# embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_SIZE))
#
# for word, i in tokenizer.word_index.items():
#     if word in model_w2v:
#         embedding_vector = model_w2v[word]
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#
# def sentence_generator(X, embeddings, batch_size):
#     while True:
#         num_recs  = X.shape[0]
#         indices = np.random.permutation(np.arange(num_recs))
#         num_batches = num_recs // batch_size
#         for bid in range(num_batches):
#             sids = indices[bid * batch_size : (bid + 1) * batch_size]
#             Xbatch = embeddings[X[sids, :]]
#             yield Xbatch, Xbatch
# train_size = 0.7
# Xtrain, Xtest = train_test_split(x_train_seq, train_size=train_size)
# train_gen = sentence_generator(Xtrain, embedding_matrix, BATCH_SIZE)
# test_gen = sentence_generator(Xtest, embedding_matrix, BATCH_SIZE)
#
# #####################################################################
# # string = ['viet nam vo dich AFF CUP 2018']
# #
# # # tokenizer_string = Tokenizer(num_words=100)
# # tokenizer.fit_on_texts(string)
# # sequences_string = tokenizer.texts_to_sequences(string)
# # string_seq = pad_sequences(sequences_string, maxlen=SEQUENCE_LEN)
# # string_embedding_matrix = np.zeros((SEQUENCE_LEN, EMBED_SIZE))
# #
# # for word, i in tokenizer.word_index.items():
# #     if word in model_w2v:
# #         embedding_vector = model_w2v[word]
# #         if embedding_vector is not None:
# #             string_embedding_matrix[i] = embedding_vector
# # string_gen = sentence_generator(string_seq,string_embedding_matrix,1)
#
# ########################################################################
# inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
# encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum",
#     name="encoder_lstm")(inputs)
# decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
# decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True),
#     merge_mode="sum",
#     name="decoder_lstm")(decoded)
#
# autoencoder = Model(inputs, decoded)
#
# autoencoder.compile(optimizer="sgd", loss="mse")
# #
# num_train_steps = len(Xtrain) // BATCH_SIZE
# num_test_steps = len(Xtest) // BATCH_SIZE
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
# checkpoint = ModelCheckpoint(filepath=model_autoencoder, save_best_only=True)
# autoencoder.fit_generator(train_gen,
#     steps_per_epoch=num_train_steps,
#     epochs=NUM_EPOCHS,
#     validation_data=test_gen,
#     validation_steps=num_test_steps,
#     callbacks=[checkpoint,early_stopping])
# # autoencoder = load_model(model_autoencoder)
# encoder = Model(autoencoder.input, autoencoder.get_layer("encoder_lstm").output)
#
# # xtest, ytest = string_gen.__next__()
# # a  = encoder.predict(ytest)
# # print(a[0].shape, a[0])
# def compute_cosine_similarity(x, y):
#     return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))
# k = 10
# cosims = np.zeros((k))
# i = 0
# for bid in range(num_test_steps):
#     xtest, ytest = test_gen.__next__()
#     print("xtest :",xtest.shape)
#     print("ytest :",ytest.shape)
#     ytest_ = autoencoder.predict(xtest)
#     print("ytest : ",ytest_.shape)
#     Xvec = encoder.predict(ytest)
#     print("Xvec :",Xvec.shape)
#     Yvec = encoder.predict(ytest_)
#     print("Yvec :",Yvec.shape)
#     for rid in range(Xvec.shape[0]):
#         if i >= k:
#             break
#         cosims[i] = compute_cosine_similarity(Xvec[rid], Yvec[rid])
#         if i <= 10:
#             print("cosine ",cosims[i])
#             i += 1
#     if i >= k:
#         break

from src.modules.autoencoder import *

with open(model_w2v_file, 'rb') as f:
    model_w2v = pickle.load(f)

train_gen,test_gen,num_train_steps,num_test_steps = pre_process_autoencoder(model_w2v)

train_model_autoencoder(train_gen,test_gen,num_train_steps,num_test_steps)
# autoencoder, encoder = load_model_encoder()
#
# evaluate_test(autoencoder,encoder,test_gen,num_test_steps)

# string1 = ['Samsung']
#
# vec1 = get_vector_sentence(string1,model_w2v,encoder)
# string2 = ['Galaxy S9']
#
# vec2 = get_vector_sentence(string2,model_w2v,encoder)
#
# print(cosine_similarity_vector(vec1,vec2))