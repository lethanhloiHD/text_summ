file_data_train = 'data/data_train/data_100.csv'
directory_train = 'data/data_train'
file_data_test = 'data/data_test/news/Summarization_BinhPT_Training.csv'

file_test_text = 'data/data_test/wiki/text/file1.txt'
file_test_sum = 'data/data_test/wiki/sum/sum.txt'

file_model_tfidf = 'models/model_tfidf.pkl'
file_feature_names='models/feature_names.pkl'

model_d2v_file = 'models/d2v.model'
model_w2v_file = 'models/w2v.model'

stopword_path = 'data/stoplist/stopwords.txt'

pattern = ['[^-!?,"]+']
mapping = {
    "tp. ":"tp.",
    "mr. ":"mr.",
    "ms. ":"ms."
}

"""
Configuration file for skip thought
"""

VOCAB_SIZE = 20000
USE_CUDA = False
DEVICES = [2]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 30

loc="models/saved_models/skip-best"
data_skip = 'data/data_skipthought/dummy_corpus.txt.pkl'