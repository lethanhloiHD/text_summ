from src.LexRank import *
from src.modules.doc2vec import *
import nltk
import  numpy as np
# import os
# from skip_thoughts.evaluate import *
# from skip_thoughts.train import *
# data_test, len = load_data_test()
# # print(data_test,len)
# avg_rouge(data_test,len)


# data = load_data_train(directory_train)
# tf= Tfidf()
# tf.build_model_featureNames(data)
# print(len(data))
# d2v = D2V()
# d2v.build_model(data)

# model = d2v.load_model()

# w2v = W2V()
# w2v.build_model_w2v(data)
# text ="""Trong các bữa tiệc của hai nhà hàng ở Hà Nội, cơm và thịt bò là thứ thường bị bỏ lại nhiều.
# Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm lượng thực phẩm dư thừa tại hai nhà hàng Le Beaulieu và Spices Garden (thuộc khách sạn Sofitel Legend Metropole Hà Nội). Theo đó, cơm là thực phẩm phải bỏ đi nhiều nhất, tính về khối lượng, còn tính về giá trị thì thịt bò đứng đầu danh sách.
# Để ghi nhận số thực phẩm bị lãng phí, nhà hàng sử dụng hệ thống bao gồm một bộ cân kỹ thuật số và máy tính bảng được kết nối với dữ liệu đám mây. Sau đó, thuật toán phân tích dữ liệu sẽ đánh giá lượng thực phẩm bỏ đi, trên cả phương diện chi phí và tác động đối với môi trường. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn.
# Nhờ đó, hai nhà hàng trên đã tiết kiệm được 11,2 tấn thức ăn – tương đương với 28.000 suất ăn sau một năm rưỡi đo lường liên tục và điều chỉnh hoạt động trong bếp. Việc hạn chế lãng phí thực phẩm cũng đồng thời tác động đến kết quả kinh doanh khách sạn, giúp tiết kiệm hơn một tỷ đồng (tương đương 44.000 USD).
# Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ mới này."""
#
# summ = """Trong các bữa tiệc của hai nhà hàng (thuộc khách sạn Sofitel Legend Metropole Hà Nội), cơm và thịt bò là thứ thường bị bỏ lại nhiều nhất. Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm thực phẩm dư thừa. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn. Sau một năm rưỡi, hai nhà hàng trên tiết kiệm được 11,2 tấn thức ăn, giúp tiết kiệm hơn 1 tỷ đồng. Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ này."""
# lexrank = LexRank(text)
# sum_lexrank = lexrank.summary()
# print(sum_lexrank)
# print(lexrank.evalutor_rouge(sum_lexrank,summ))

# load_data_train_skipthought(directory_train)
# train_skipthought()
# usable_encoder = UsableEncoder()
# X = [
#     'viet nam vo dich',
# ]
# print(usable_encoder.encode(X))

