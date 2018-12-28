import operator
import re
import networkx as nx
from src.modules.tfidf import *
from src.modules.word2vec import *
from src.modules.doc2vec import *
from src.modules.autoencoder import *
from nltk.tokenize import RegexpTokenizer
from pyvi import ViTokenizer
from rouge import Rouge
import requests
from src.modules.lda import *

tokenizer = RegexpTokenizer(r'\w+')


def mmr_score(graph, sentence, selected_sentence, lamda=0.7):
    sims = []
    for sent in selected_sentence:
        if sent[0] in graph.nodes() and sentence[0] in graph.nodes():
            try:
                weight = graph.get_edge_data(sentence[0], sent[0])['weight']
            except:
                weight = 0
            sims.append(weight)
    max_sim = max(sims)
    score = lamda * sentence[1] - (1 - lamda) * max_sim
    return score


def mmr_ranking(graph, ranked_sentences, limit_mmr):
    num_sentences = len(ranked_sentences)
    limit_sentences = math.ceil(num_sentences * limit_mmr)
    selected_sentences = [ranked_sentences[0]]
    candidate_sentences = ranked_sentences[1:num_sentences]

    while len(selected_sentences) < limit_sentences:
        sentence_ = ()
        score_ = mmr_score(graph, candidate_sentences[0], selected_sentences)

        for sentence in candidate_sentences:
            score = mmr_score(graph, sentence, selected_sentences)
            if score >= score_:
                sentence_ = sentence
        selected_sentences.append(sentence_)
        candidate_sentences.remove(sentence_)
    return selected_sentences


def build_graph(text, similarityThreshold=0.1, max_threshold=1.0,
                tfidf_option=False,
                doc2vec_option=False,
                word2vec_option=False,
                autoencoder_option=True):
    """
    Build graph with representation of sentence with tfidf score
    :param text: data train
    :param similarityThreshold:
    :return: graph
    """
    tf = Tfidf()
    d2v = D2V()
    w2v = W2V()
    graph = nx.Graph()
    sen2index = {}
    sentences_split = split_sentences(text)

    sentences = []
    for index, sentence in enumerate(sentences_split):
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > 3:
            sentences.append(sentence)
            # print(sentence)
            sen2index.update({
                str(sentence): index
            })
    print("number sentence ", len(sentences))

    if len(sentences) > 2:
        models, feature_names = tf.load_model_featureNames()
        model_d2v = d2v.load_model()
        model_w2v = w2v.load_model()
        autoencoder, encoder = load_model_encoder()

        graph.add_nodes_from(sentences)

        if tfidf_option:
            for node1 in graph.nodes():
                for node2 in graph.nodes():
                    weight_value = tf.cosine_similarity_tfidf(models, node1, node2)[0][0]
                    if similarityThreshold < weight_value < max_threshold:
                        graph.add_edge(node1, node2, weight=weight_value)

        elif doc2vec_option:
            for node1 in graph.nodes():
                for node2 in graph.nodes():
                    weight_value = d2v.get_cosine_similary(model_d2v, node1, node2)
                    if similarityThreshold < weight_value < max_threshold:
                        graph.add_edge(node1, node2, weight=weight_value)

        elif word2vec_option:
            for node1 in graph.nodes():
                for node2 in graph.nodes():
                    weight_value = w2v.get_cosine_similary_w2v_tfidf(tf, models, feature_names, model_w2v, node1, node2)
                    if similarityThreshold < weight_value < max_threshold:
                        graph.add_edge(node1, node2, weight=weight_value)

        elif autoencoder_option:
            for node1 in graph.nodes():
                for node2 in graph.nodes():
                    try :
                        weight_value = get_cosine_simi_autoencoder(node1, node2, model_w2v, encoder)
                        if similarityThreshold < weight_value < max_threshold:
                            graph.add_edge(node1, node2, weight=weight_value)
                    except :
                        print(" cannot cacl similar sentence ")


    return graph, sen2index


def pre_process(text):
    text_token = requests.post(url=url_token, data={"text": text}).text
    text_ = pre_process_text(text_token, remove_number_punctuation=False)
    text_ = text_.lower()
    return text_


def avg_rouge(data_test):
    len = 0
    result = {}
    rouge_1_f1, rouge_1_p, rouge_1_r, rouge_2_f1, rouge_2_r, rouge_2_p, = 0, 0, 0, 0, 0, 0

    for row in data_test:
        text, summ = row['text'], row['summ']
        lr = Rank(text)
        try :
            summ_lexrank, summ_sents = lr.summary()
            score = lr.evaluation_rouge(summ_lexrank, summ)
            print(score)
            rouge_1_f1 += float(score[0]['rouge-1']['f'])
            rouge_1_p += float(score[0]['rouge-1']['p'])
            rouge_1_r += float(score[0]['rouge-1']['r'])
            rouge_2_f1 += float(score[0]['rouge-2']['f'])
            rouge_2_p += float(score[0]['rouge-2']['p'])
            rouge_2_r += float(score[0]['rouge-2']['r'])
            len +=1
        except :
            print("cannot calc score !")

    r1_f1, r2_f1, r1_r, r2_r, r1_p, r2_p = float(rouge_1_f1 / int(len)), float(rouge_2_f1 / int(len)), float(
        rouge_1_r / int(len)), float(rouge_2_r / int(len)), float(rouge_1_p / int(len)), float(rouge_2_p / int(len))

    result.update({"rouge1-f1": r1_f1, "rouge1-r": r1_r, "rouge1-p": r1_p,
                   "rouge2-f1": r2_f1, "rouge2-r": r2_r, "rouge2-p": r2_p})

    print("rouge1-f1", r1_f1, "rouge1-r", r1_r, "rouge1-p", r1_p, "rouge2-f1", r2_f1, "rouge2-r", r2_r, "rouge2-p",
          r2_p)
    return result


class Rank(object):

    def __init__(self, text,
                 tfidf_option=False,
                 doc2vec_option=False,
                 word2vec_option=False,
                 autoencoder_option =True):
        if not tfidf_option and not doc2vec_option and not word2vec_option and not autoencoder_option :
            tfidf_option = True
        self.text = pre_process(text)
        self.graph, self.sent2id = build_graph(self.text,
                                               tfidf_option=tfidf_option,
                                               doc2vec_option=doc2vec_option,
                                               word2vec_option=word2vec_option,
                                               autoencoder_option=autoencoder_option)

    def summary(self, limit_pagerank = 0.8,
                limit_mmr=1.0,
                threshold_number_word=150,
                using_postion_score = True,
                option_mmr=True,
                option_plmmr=False):

        summ_sents = {}
        summary_sentences = ""
        total_sentences = len(self.graph.nodes())
        if total_sentences > 2:
            n_sentences = int(total_sentences * limit_pagerank)
            print(len(self.graph.nodes()))
            rankings = nx.pagerank(self.graph, alpha = 0.7)
            print("rankings : ", rankings)
            """"" Using score position combine with score pagerank for ranking sentence in document """
            if using_postion_score:
                for sent, score_pr in rankings.items():
                    score_position = float(score_position_sentence(total_sentences, self.sent2id[sent]))
                    score_final = 0.8 * score_pr + 0.2 * score_position
                    rankings.update({
                        sent: score_final
                    })
            ranked_sentences = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            ranked_sentences = ranked_sentences[:n_sentences]
            print("ranking_sentences : ", ranked_sentences)
            if option_mmr:
                select_sents = mmr_ranking(self.graph, ranked_sentences, limit_mmr)
                select_sentences = [sent[0] for sent in select_sents]

            elif option_plmmr:
                candidate_ranked = [sent[0] for sent in ranked_sentences]
                # print("candidate_ranked plmmr", candidate_ranked)
                select_sentences = plMMR(self.text, candidate_ranked, math.ceil(n_sentences * limit_mmr))
            else:
                select_sentences = [sent[0] for sent in ranked_sentences]
            total_tokens = 0
            # print("select_sentences ", select_sentences)

            if len(select_sentences) > 0:
                for item in select_sentences:
                    # tokens = tokenizer.tokenize(item[0].replace("_", " "))
                    tokens = tokenizer.tokenize(item)
                    num_tokens = len(tokens)
                    total_tokens += num_tokens

                    if total_tokens <= threshold_number_word:
                        selected_sen = item
                        print("item", item)
                        summ_sents.update({
                            str(selected_sen): self.sent2id[selected_sen]
                        })
                        print("totals tokens :", total_tokens)

            summ_sents = sorted(summ_sents.items(), key=operator.itemgetter(1))
            print("lenght sum :", len(summ_sents))
            summary_sentences = " ".join(sent[0] for sent in summ_sents)
            print("summary sentences : ", summary_sentences)
        return summary_sentences, summ_sents

    def evaluation_rouge(self, summ_lexrank, summ):
        evaluator = Rouge()
        summ = requests.post(url=url_token, data={"text": summ}).text
        score = evaluator.get_scores(summ_lexrank, summ)
        return score
