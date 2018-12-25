from Util.utility import *
from src.Graph_base import *
from src.modules.tfidf import *
from src.modules.word2vec import *
from src.modules.doc2vec import *
from src.modules.autoencoder import *
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from src.modules.lda import *

def pre_process_cluster(text):
    text_token = requests.post(url=url_token, data={"text": text}).text
    text_ = pre_process_text(text_token, remove_number_punctuation=False)
    text_ = text_.lower()
    return text_


def avg_rouge_cluster(data_test):
    len = 0
    result = {}
    rouge_1_f1, rouge_1_p, rouge_1_r, rouge_2_f1, rouge_2_r, rouge_2_p, = 0, 0, 0, 0, 0, 0

    for row in data_test:
        text, summ = row['text'], row['summ']
        cluster = Cluster(text)
        try :
            summ_cluster = cluster.get_summary()
            score = cluster.evaluation_rouge(summ_cluster, summ)
            print(score)
            rouge_1_f1 += float(score[0]['rouge-1']['f'])
            rouge_1_p += float(score[0]['rouge-1']['p'])
            rouge_1_r += float(score[0]['rouge-1']['r'])
            rouge_2_f1 += float(score[0]['rouge-2']['f'])
            rouge_2_p += float(score[0]['rouge-2']['p'])
            rouge_2_r += float(score[0]['rouge-2']['r'])
            len +=1
        except :
             print("Can not calc score !")

    r1_f1, r2_f1, r1_r, r2_r, r1_p, r2_p = float(rouge_1_f1 / int(len)), float(rouge_2_f1 / int(len)), \
                                                   float(rouge_1_r / int(len)), float(rouge_2_r / int(len)), \
                                                   float(rouge_1_p / int(len)),float(rouge_2_p / int(len))


    result.update({"rouge1-f1": r1_f1, "rouge1-r": r1_r, "rouge1-p": r1_p,
                   "rouge2-f1": r2_f1, "rouge2-r": r2_r, "rouge2-p": r2_p})

    print("rouge1-f1", r1_f1, "rouge1-r", r1_r, "rouge1-p", r1_p, "rouge2-f1", r2_f1, "rouge2-r", r2_r, "rouge2-p",r2_p)

    return result


class Cluster(object):

    def __init__(self, text,
                 ratio = 0.5,
                 tfidf_option=False,
                 doc2vec_option=False,
                 word2vec_option=False,
                 autoencoder_option=True):

        self.text = pre_process_cluster(text)
        self.sentences, self.embedding_sentences = self.get_embedding_sentence(tfidf_option=tfidf_option,
                                                                               doc2vec_option=doc2vec_option,
                                                                               word2vec_option=word2vec_option,
                                                                               autoencoder_option=autoencoder_option)
        self.ratio_cluster = ratio

    def get_embedding_sentence(self,
                               tfidf_option=False,
                               doc2vec_option=False,
                               word2vec_option=False,
                               autoencoder_option=False):
        if not tfidf_option and not doc2vec_option \
           and not word2vec_option and not autoencoder_option :
            tfidf_option = True

        tf = Tfidf()
        model_tfidf, feature_names = tf.load_model_featureNames()
        d2v = D2V()
        model_d2v = d2v.load_model()
        w2v = W2V()
        model_w2v = w2v.load_model()
        autoencoder, encoder = load_model_encoder()

        sentences_split = split_sentences(self.text)
        sentences = []
        embedding_sentences = []
        for sentence in sentences_split:
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > 1:
                sentences.append(sentence)
        if len(sentences) > 2:
            if tfidf_option:
                embedding_sentences = model_tfidf.fit_transform(sentences)
            elif doc2vec_option:
                for sent in sentences:
                    vector = d2v.get_vector_sentences(model_d2v, sent)
                    embedding_sentences.append(vector)
            elif word2vec_option:
                for sent in sentences:
                    vector = w2v.avg_representation_w2v_tfidf(tf,model_tfidf,feature_names,model_w2v,sent)
                    embedding_sentences.append(vector)
            elif autoencoder_option:
                for sent in sentences:
                    try :
                        vector = get_vector_sentence(sent,model_w2v,encoder=encoder)
                        embedding_sentences.append(vector)
                    except :
                        print("Cannot find embedding of sentence !")

        return sentences, embedding_sentences

    def get_summary(self, plmmr_option = False , limit_plmmr = 0.8):
        num_cluster = int(math.ceil(self.ratio_cluster * len(self.sentences)))
        print("number cluster :", len(self.sentences),num_cluster)
        kmeans = KMeans(n_clusters=num_cluster, init='k-means++', max_iter= 300)
        kmeans.fit(self.embedding_sentences)
        avg = []
        for i in range(num_cluster):
            idx = np.where(kmeans.labels_ == i)[0]
            avg.append(np.mean(idx))

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, self.embedding_sentences)
        ordering = sorted(range(num_cluster), key=lambda k: avg[k])
        index = [closest[idx] for idx in ordering]
        index.sort(reverse=False)
        summary_sentences = [self.sentences[idx] for idx in index]
        if plmmr_option :
            limit = math.ceil(limit_plmmr * len(summary_sentences))
            sents = plMMR(self.text,summary_sentences,limit_sentences= limit)
            summary_sentences = sents
        for sent in summary_sentences :
            print("summary",sent)
        summary = ' '.join(sent for sent in summary_sentences)
        return summary

    def evaluation_rouge(self, summay_cluster, summ):
        evaluator = Rouge()
        summ = requests.post(url=url_token, data={"text": summ}).text
        score = evaluator.get_scores(summay_cluster, summ)
        print("score ", score)

        return score
