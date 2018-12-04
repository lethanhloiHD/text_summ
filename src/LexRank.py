import operator
import re
import networkx as nx
from src.modules.tfidf import *
from src.modules.word2vec import *
from src.modules.doc2vec import *
# from TextGraphics.src.graph import TextGraph
# import rouge
from nltk.tokenize import RegexpTokenizer
from  pyvi import ViTokenizer
from rouge import Rouge

tokenizer = RegexpTokenizer(r'\w+')

def mmr_score(graph , sentence,selected_sentence, lamda = 0.7) :
    sims = []
    for sent in selected_sentence :
        if sent[0] in graph.nodes() and sentence[0] in graph.nodes():
            try :
                weight = graph.get_edge_data(sentence[0],sent[0])['weight']
            except :
                weight = 0
            sims.append(weight)
    max_sim = max(sims)
    score = lamda * sentence[1] - (1 - lamda) * max_sim
    return score

def mmr_ranking(graph, ranked_sentences,limit_mmr):
    num_sentences = len(ranked_sentences)
    limit_sentences = math.ceil(num_sentences * limit_mmr)
    selected_sentences = [ranked_sentences[0]]
    candidate_sentences = ranked_sentences[1:num_sentences]

    while len(selected_sentences) < limit_sentences:
        sentence_ = ()
        score_ = mmr_score(graph,candidate_sentences[0],selected_sentences)

        for sentence in candidate_sentences :
            score  = mmr_score(graph,sentence,selected_sentences)
            if score >= score_ :
                sentence_ = sentence
        selected_sentences.append(sentence_)
        candidate_sentences.remove(sentence_)
    return selected_sentences


def build_graph(text, similarityThreshold = 0.001, max_threshold = 1.0,
                tfidf_option= True, doc2vec_option = False,word2vec_option = False):
    """
    Build graph with representation of sentence with tfidf score
    :param text: data train
    :param similarityThreshold:
    :return: graph
    """
    tf = Tfidf()
    d2v = D2V()
    w2v = W2V()
    sentences_split = split_sentences(text)
    sentences = []
    for sentence in sentences_split :
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > 2 :
            sentences.append(sentence)

    print("number sentence ", len(sentences))
    models, feature_names = tf.load_model_featureNames()
    model_d2v = d2v.load_model()
    model_w2v = w2v.load_model()
    graph = nx.Graph()
    graph.add_nodes_from(sentences)

    if tfidf_option:
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                weight_value = tf.cosine_similarity_tfidf(models, node1, node2)[0][0]
                if weight_value > similarityThreshold and weight_value < max_threshold:
                    graph.add_edge(node1, node2, weight=weight_value)

    elif doc2vec_option:
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                weight_value = d2v.get_cosine_similary(model_d2v,node1,node2)
                if weight_value > similarityThreshold and weight_value < max_threshold:
                    graph.add_edge(node1, node2, weight=weight_value)

    elif word2vec_option :
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                weight_value = w2v.get_cosine_similary_w2v_tfidf(tf,models,feature_names,model_w2v,node1, node2)
                if weight_value > similarityThreshold and weight_value < max_threshold:
                    graph.add_edge(node1, node2, weight=weight_value)
    return  graph

def pre_process(text):
    text_token = ViTokenizer.tokenize(text)
    text_ = pre_process_data(text_token, remove_number_punctuation= False)
    text_ = text_.lower()
    return text_


def avg_rouge(data_test, len):
    rouge_1_f1,rouge_1_p,rouge_1_r,rouge_2_f1,rouge_2_r,rouge_2_p, = 0,0,0,0,0,0

    for row in data_test:
        text = row['text']
        summ = row['summ']
        lr = LexRank(text)
        summ_lexrank =  lr.summary()
        score = lr.evalutor_rouge(summ_lexrank,summ)
        print(score)
        rouge_1_f1 += float(score[0]['rouge-1']['f'])
        rouge_1_p  += float(score[0]['rouge-1']['p'])
        rouge_1_r += float(score[0]['rouge-1']['r'])
        rouge_2_f1 += float(score[0]['rouge-2']['f'])
        rouge_2_p += float(score[0]['rouge-2']['p'])
        rouge_2_r += float(score[0]['rouge-2']['r'])

    r1_f1 = float(rouge_1_f1/int(len))
    print("r1_f1 : ",r1_f1)
    r2_f1 = float(rouge_2_f1/int(len))
    print("r2_f1 : ",r2_f1)

    r1_r = float(rouge_1_r / int(len))
    print("r1_r : ",r1_r)
    r2_r = float(rouge_2_r / int(len))
    print("r2_r : ",r2_r )

    r1_p = float(rouge_1_p / int(len))
    print("r1_p : ",r1_p)
    r2_p = float(rouge_2_p / int(len))
    print("r2_p : ",r2_p)


    return r1_f1,r2_f1

class LexRank:

    def __init__(self, text):
        self.text = pre_process(text)
        self.graph = build_graph(self.text, tfidf_option=False,
                                 doc2vec_option= False, word2vec_option=True)


    def summary(self, limit_pagerank=0.5, limit_mmr = 0.6, number_token = 150):
        total_sentences = len(self.graph.nodes())
        n_sentences = int(total_sentences * limit_pagerank)
        print(len(self.graph.nodes()), math.ceil(n_sentences * limit_mmr))
        rankings = nx.pagerank(self.graph, alpha=0.85)
        ranked_sentences = sorted(rankings.items(), key= lambda x : x[1] , reverse=True)

        summary_sentences = ""
        ranked_sentences = ranked_sentences[:n_sentences]
        select_sentences = mmr_ranking(self.graph, ranked_sentences, limit_mmr)
        for item in select_sentences :
            summary_sentences += item[0]+". "

        return summary_sentences


    def evalutor_rouge(self,summ_lexrank, summ):
        evaluator = Rouge()
        summ = ViTokenizer.tokenize(summ)
        score = evaluator.get_scores(summ_lexrank,summ)
        return score
