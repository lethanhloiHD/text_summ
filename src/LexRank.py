import operator
import re
import networkx as nx
from src.modules.tfidf import *
from src.modules._word2vec_ import *
from src.modules._doc2vec_ import *
# from TextGraphics.src.graph import TextGraph
# import rouge
from  pyvi import ViTokenizer
from rouge import Rouge


def mmr_score(graph , sentence,selected_sentence, lamda = 0.7) :
    sims = []
    for sent in selected_sentence :
        if sent[0] in graph.nodes() and sentence[0] in graph.nodes():
            weight = graph.get_edge_data(sentence[0],sent[0])['weight']
            print("weight mmr", weight)
            sims.append(weight)
    max_sim = max(sims)
    score = lamda * sentence[1] - (1 - lamda) * max_sim
    return score

def mmr_ranking(graph, ranked_sentences,limit_mmr):
    num_sentences = len(ranked_sentences)
    limit_sentences = int(num_sentences * limit_mmr)
    print(num_sentences,limit_sentences)
    selected_sentences = [ranked_sentences[0]]
    print(selected_sentences)
    candidate_sentences = ranked_sentences[1:num_sentences]
    print(candidate_sentences)
    while len(selected_sentences) < limit_sentences:
        sentence_ = ()
        score_ = mmr_score(graph,candidate_sentences[0],selected_sentences)
        print("score mmr :", score_)
        for sentence in candidate_sentences :
            score  = mmr_score(graph,sentence,selected_sentences)
            if score >= score_ :
                sentence_ = sentence
        selected_sentences.append(sentence_)
        candidate_sentences.remove(sentence_)
    return selected_sentences


def build_graph(text, similarityThreshold = 0.01, max_threshold = 0.95,
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
    sentences = split_sentences(text)
    print("number sentence ", len(sentences))
    models, feature_names = tf.load_model_featureNames()
    model_d2v = d2v.load_model()
    graph = nx.Graph()
    graph.add_nodes_from(sentences)

    if tfidf_option:
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                weight_value = tf.cosine_similarity_tfidf(models, node1, node2)[0][0]
                if weight_value > similarityThreshold and weight_value < max_threshold:
                    print("weight ", weight_value)
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
                weight_value = w2v.get_cosine_similary_w2v_tfidf(models,node1, node2)
                if weight_value > similarityThreshold and weight_value < max_threshold:
                    graph.add_edge(node1, node2, weight=weight_value)
    return  graph


class LexRank:
    def __init__(self, text):
        # self.text = ViTokenizer.tokenize(text)
        self.text  = text
        self.graph = build_graph(self.text, tfidf_option=False,doc2vec_option= False, word2vec_option=True)


    def summary(self, limit_pagerank=0.8, limit_mmr = 0.6):
        total_sentences = len(self.graph.nodes())
        n_sentences = int(total_sentences * limit_pagerank)
        print(len(self.graph.nodes()), (n_sentences))
        rankings = nx.pagerank(self.graph, alpha=0.85)
        ranked_sentences = sorted(rankings.items(), key= lambda x : x[1] , reverse=True)

        summary_sentences = ""
        ranked_sentences = ranked_sentences[:n_sentences]
        select_sentences = mmr_ranking(self.graph, ranked_sentences, limit_mmr)
        for item in select_sentences :
            print(item)
            summary_sentences += item[0]

        return summary_sentences


    def evalutor_rouge(self,text, summ):
        evaluator = Rouge()
        score = evaluator.get_scores(text,summ)
        return score
