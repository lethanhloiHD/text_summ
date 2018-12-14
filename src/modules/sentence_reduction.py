from Util.utility import *
from pyvi import ViTokenizer,ViPosTagger
import requests


def count_doc_occur_topicWord(data_train,topic_words):
    counts = {}
    for word in topic_words :
        number_doc_occur = 0
        word = word.lower()
        for row in data_train :
            if word in row :
                number_doc_occur +=1
        counts.update({word: number_doc_occur})
    return counts

def count_occur_sentence(setence):
    setence = ViTokenizer.tokenize(setence)
    topic_words, other_words = get_topic_word_in_sentence(setence)

    number_topicwords_sent = {}
    for tw in topic_words :
        count = setence.count(tw)
        number_topicwords_sent.update({
            tw.lower() : int(count)
        })
        print(tw, count)
    return topic_words,other_words,number_topicwords_sent


def information_significant_score(train_set, original_sentence, ratio_reduce_topicWords = 1.0):
    total_doc_train = len(train_set)
    topic_words,other_words, number_topicwords_sent = count_occur_sentence(setence= original_sentence)
    total_topic_words = len(topic_words)
    counts_doc_occur = count_doc_occur_topicWord(train_set,topic_words)

    print("number_topicwords_sent :" ,number_topicwords_sent)
    print("total_topic_words :", total_topic_words)
    print("counts_doc_occur :", counts_doc_occur)

    info_score = {}
    for tw in topic_words:
        tw = tw.lower()
        Ns = number_topicwords_sent[tw]
        Nd = counts_doc_occur[tw]
        score = float(Ns/ total_topic_words) + float(Nd/total_doc_train)
        info_score.update({ tw : score})
    info_score = sorted(info_score.items(), key= lambda  x : x[1],
                        reverse= True)[:math.ceil(ratio_reduce_topicWords * total_topic_words)]

    for i,v in info_score:
        print(i,v)
    return info_score, other_words


def get_topic_word_in_sentence(sentence):
    topic_words, other_words = [],[]
    # token = list(ViTokenizer.tokenize(setence).split())
    token = list(requests.post(url=url_token, data={"text": sentence}).text.split())
    tokens = " ".join((t for t in token if len(t) > 1))
    token_pos = ViPosTagger.postagging(tokens)
    for i in range(len(token_pos[0])):
        # print(token_pos[0][i], token_pos[1][i])
        if token_pos[1][i] in noun :
            topic_words.append(token_pos[0][i])
        else :
            other_words.append(token_pos[0][i])
    return topic_words,other_words




