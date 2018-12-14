from src.Cluster import *
# sentence = 'Nguyễn_Tấn_Dũng tham_dự'
#
# tf= Tfidf()
# models_tfidf,feature = tf.load_model_featureNames()
#
# n_grams = word_grams(sentence.lower())
# print(n_grams)
# print(tf.get_tfidf_word_in_sentence(models_tfidf,feature,sentence))

# # data_train = load_data_train(directory_train)
#
# string = "Trong tháng 4, Thủ tướng Việt Nam Nguyễn Tấn Dũng sẽ tham dự Hội nghị Thượng đỉnh về " \
#          "an toàn hạt nhân do Tổng thống Mỹ Barack Obama chủ trì tại Washington."
#
# a = word_grams(string)
# to = get_topic_word_in_sentence(string)
# print(to)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import pairwise_distances_argmin_min
#
# documents = ["This little kitty came to play when I was eating at a restaurant.",
#              "Merley has the best squooshy kitten belly.",
#              "Google Translate app is incredible.",
#              "If you open 100 tab in google you get a smiley face.",
#              "Best cat photo I've ever taken.",
#              "Climbing ninja cat.",
#              "Impressed with google map feedback.",
#              "Key promoter extension for Google Chrome."]
#
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents)
#
# true_k = 3
# kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=10, n_init=1)
# kmeans.fit(X)
#
# print("Top terms per cluster:")
# # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# # terms = vectorizer.get_feature_names()
# avg = []
# for i in range(true_k):
#     idx = np.where(kmeans.labels_ == i)[0]
#     print(idx)
#     avg.append(np.mean(idx))
#
# closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
# ordering = sorted(range(true_k), key=lambda k: avg[k], reverse= True)
# summary = ' '.join([documents[closest[idx]] for idx in ordering])
#
# print(summary)
data_test, len = load_data_test()

avg_rouge_cluster(data_test,len)

# text = data_test[0]['text']
# summ = data_test[0]['summ']
# print(text)
# cluster = Cluster(text)
# sentences, emb_sentences  = cluster.get_embedding_sentence()
# summ_cluster = cluster.get_summary(sentences, emb_sentences)
# cluster.evaluation_rouge(summ_cluster, summ)
