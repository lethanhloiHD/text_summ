from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
import os
import time
from src.modules.lda import *

string = "viet nam ... vo dich"
print(string.replace("..."," "))

# data = load_data_train(directory_train)
# #
# # corpus, dic = pre_process_lda(data)
# #
# lda = LDA()
# lda.build_model_lda(data)

# model_lda , dictionary_lda = lda.load_model_lda()
#
# lda.get_vector_lda("Samsung sắp đưa tân b Galaxy A6s ra thị trường: mặt lưng bóng bẩy, dùng màn LCD 6 inch, vị trí đặt cảm biến vân tay hơi cao")
# lda.get_vector_lda("Thoạt nhìn, vị trí cảm biến vân tay gần như ngang bằng với cụm camera sau, do đó thao tác mở máy có thể sẽ gặp khó khăn với những người tay ngắn")