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
# data_test = load_data_test()
#
# avg_rouge_cluster(data_test)
# text ="""Trong các bữa tiệc của hai nhà hàng ở Hà Nội, cơm và thịt bò là thứ thường bị bỏ lại nhiều.
# Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm lượng thực phẩm dư thừa tại hai nhà hàng Le Beaulieu và Spices Garden (thuộc khách sạn Sofitel Legend Metropole Hà Nội). Theo đó, cơm là thực phẩm phải bỏ đi nhiều nhất, tính về khối lượng, còn tính về giá trị thì thịt bò đứng đầu danh sách.
# Để ghi nhận số thực phẩm bị lãng phí, nhà hàng sử dụng hệ thống bao gồm một bộ cân kỹ thuật số và máy tính bảng được kết nối với dữ liệu đám mây. Sau đó, thuật toán phân tích dữ liệu sẽ đánh giá lượng thực phẩm bỏ đi, trên cả phương diện chi phí và tác động đối với môi trường. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn.
# Nhờ đó, hai nhà hàng trên đã tiết kiệm được 11,2 tấn thức ăn – tương đương với 28.000 suất ăn sau một năm rưỡi đo lường liên tục và điều chỉnh hoạt động trong bếp. Việc hạn chế lãng phí thực phẩm cũng đồng thời tác động đến kết quả kinh doanh khách sạn, giúp tiết kiệm hơn một tỷ đồng (tương đương 44.000 USD)
# Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ mới này."""
#
# summ = """Trong các bữa tiệc của hai nhà hàng (thuộc khách sạn Sofitel Legend Metropole Hà Nội), cơm và thịt bò là thứ thường bị bỏ lại nhiều nhất. Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm thực phẩm dư thừa. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn. Sau một năm rưỡi, hai nhà hàng trên tiết kiệm được 11,2 tấn thức ăn, giúp tiết kiệm hơn 1 tỷ đồng. Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ này."""
data_test= load_data_test()
text = data_test[0]['text']
summ = data_test[0]['summ']
# print(text)
cluster = Cluster(text)
sentences, emb_sentences  = cluster.get_embedding_sentence()
summ_cluster = cluster.get_summary()
# cluster.evaluation_rouge(summ_cluster, summ)
