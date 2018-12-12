from src.LexRank import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
import os
import time
from src.sentence_reduction import *

data_train = load_data_train(directory_train)

string = "Trong tháng 4, Thủ tướng Việt Nam Nguyễn Tấn Dũng sẽ tham dự Hội nghị Thượng đỉnh về an toàn hạt nhân do Tổng thống Mỹ Barack Obama chủ trì tại Washington"

information_significant_score(data_train,string)