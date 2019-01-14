from src.Cluster import *
from run_graph import *


def run_cluster():
    text = get_input()

    cluster = Cluster(text,
                      ratio=0.4,
                      tfidf_option=True,
                      doc2vec_option=False,
                      word2vec_option=False,
                      autoencoder_option=False
                      )
    summ_cluster = cluster.get_summary()
    set_output(summ_cluster)


if __name__ == '__main__':
    run()

