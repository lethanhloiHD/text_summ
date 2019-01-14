# directory save file train and test
directory_train = 'data/data_train'
file_data_test = 'data/data_test/news/Summarization_BinhPT_Training.csv'
file_test_text = 'data/data_test/wiki/text/file1.txt'
file_test_sum = 'data/data_test/wiki/sum/sum.txt'

# file save models
file_model_tfidf = 'models/model_tfidf.pkl'
file_feature_names = 'models/feature_names.pkl'
model_d2v_file = 'models/d2v.model'
model_w2v_file = 'models/w2v.model'
model_autoencoder = 'models/autoencoder.h5'

# lda
save_lda = 'models/save_model_lda/lda_model.pkl'
save_dictonary_lda = 'models/save_model_lda/lda_dict.pkl'
NUM_TOPICS = 50

# file save config
config_token = 'data/config/config_token.json'
stopword_path = 'data/stoplist/stopwords.txt'

# file demo
input_file = "data/data_demo/input.txt"
output_file = "data/data_demo/output.txt"

# var save config
noun = ["N", "Ny", "Np"]
pattern = ['[^-!?,"]+']
mapping = {
    "tp. ": "tp.",
    "mr. ": "mr.",
    "ms. ": "ms."
}

# Configuration file for skip thought
VOCAB_SIZE = 20000
USE_CUDA = False
DEVICES = [2]
CUDA_DEVICE = DEVICES[0]
VERSION = 1
MAXLEN = 30
loc = "models/saved_models/skip-best"
data_skip = 'data/data_skipthought/dummy_corpus.txt.pkl'
