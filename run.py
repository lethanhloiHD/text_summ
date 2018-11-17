from src.LexRank import *
from src.modules._doc2vec_ import *
import nltk
import  numpy as np
import os



# data = load_data_train(directory_train)
# print(len(data))
# d2v = D2V()
# model = d2v.load_model()
# sen1 = data[0]
# sen2 = data[1]
#
# print(d2v.get_cosine_similary(model,sen1,sen2))

# sentence1 = data[1]data
# text = load_data_test(file_test_text)
# summ = load_data_test(file_test_sum)
w2v = W2V()

text ="""Trong các bữa tiệc của hai nhà hàng ở Hà Nội, cơm và thịt bò là thứ thường bị bỏ lại nhiều.
Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm lượng thực phẩm dư thừa tại hai nhà hàng Le Beaulieu và Spices Garden (thuộc khách sạn Sofitel Legend Metropole Hà Nội). Theo đó, cơm là thực phẩm phải bỏ đi nhiều nhất, tính về khối lượng, còn tính về giá trị thì thịt bò đứng đầu danh sách.
Để ghi nhận số thực phẩm bị lãng phí, nhà hàng sử dụng hệ thống bao gồm một bộ cân kỹ thuật số và máy tính bảng được kết nối với dữ liệu đám mây. Sau đó, thuật toán phân tích dữ liệu sẽ đánh giá lượng thực phẩm bỏ đi, trên cả phương diện chi phí và tác động đối với môi trường. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn.
Nhờ đó, hai nhà hàng trên đã tiết kiệm được 11,2 tấn thức ăn – tương đương với 28.000 suất ăn sau một năm rưỡi đo lường liên tục và điều chỉnh hoạt động trong bếp. Việc hạn chế lãng phí thực phẩm cũng đồng thời tác động đến kết quả kinh doanh khách sạn, giúp tiết kiệm hơn một tỷ đồng (tương đương 44.000 USD).
Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ mới này."""


summ = """Nhờ công nghệ Winnow, hai nhà hàng 5 sao tại Hà Nội đã xác định được cơm và thịt bò thường bị bỏ lại nhiều nhất. Theo đó, cơm phải bỏ đi nhiều nhất, tính về khối lượng, còn thịt bò đứng đầu về giá trị.
Một hệ thống gồm cân kỹ thuật số và máy tính bảng kết nối với dữ liệu đám mây sẽ đánh giá lượng thực phẩm bỏ đi dựa theo chi phí và tác động đối với môi trường.
Kết quả sẽ được sử dụng để cắt giảm thực phẩm hay bị thừa và lượng thức ăn cần chế biến. Nhờ đó, hai nhà hàng trên đã cắt giảm được 11,2 tấn thức ăn, góp phần tiết kiệm hơn một tỷ đồng.
Hiện công nghệ mới này đã giúp 38 khách sạn thuộc AccorHotels cắt giảm lượng lớn chất thải thực phẩm."""
lexrank = LexRank(text)
sum_lexrank = lexrank.summary()
print(sum_lexrank)
print(lexrank.evalutor_rouge(sum_lexrank,summ))

