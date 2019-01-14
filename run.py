from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
# from skip_thoughts.evaluate import *
# from skip_thoughts.train import *

# data_test= load_data_test()
# avg_rouge(data_test)
# lexrank = Rank(text,tfidf_option=False,
#                     doc2vec_option=False,
#                     word2vec_option=False,
#                     autoencoder_option=False
#             )
# sum_lexrank = lexrank.summary(option_mmr=True, using_postion_score= True)
# print(lexrank.evaluation_rouge(sum_lexrank,summ))
import argparse

def get_input():
    with open(input_file,"r") as file:
        data = file.read()
    return data

def set_output(sentences):
    with open(output_file, "w") as file:
        # for sentence in set_sentences :
        file.write(sentences)

def run():
    # autoencoder_option = doc2vec_option = word2vec_option = tfidf_option = False
    # # parser = argparse.ArgumentParser(description='Description of your program')
    # # parser.add_argument('-o', '--model', help='Using model for sentence presentation', required=True)
    # # # parser.add_argument('-t', '--type', help='Choose graph-base or cluster-base sentences', required=True)
    # # # parser.add_argument('-r', '--ratio', help='Choose number ratio cluster for cluster-base sentences',
    # # #                     required=False, default=0.4)
    # # parser.add_argument('-m', '--option_mmr', help='Using MMR for select sentence',
    # #                     required=False,default=True)
    # # parser.add_argument('-p', '--postion_score', help='Using Position_Score of sentence in document',
    # #                     required=False, default= True)
    # #
    # # args = (parser.parse_args())
    # # if args.model == 'tf':
    # #     tfidf_option = True
    # # elif args.model == 'w2v':
    # #     word2vec_option = True
    # # elif args.model == 'd2v':
    # #     doc2vec_option = True
    # # elif args.model == 'ae':
    # #     autoencoder_option = True

    text = get_input()
    pre_process(text)

    lexrank = Rank(text,tfidf_option=True,
                        doc2vec_option=False,
                        word2vec_option=False,
                        autoencoder_option=False
                )
    sum_lexrank = lexrank.summary(option_mmr=True, using_postion_score= True)
    set_output(sum_lexrank)

if __name__ == '__main__':
    run()




