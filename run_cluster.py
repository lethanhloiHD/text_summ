from src.Cluster import *

# data_test = load_data_test()
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
cluster = Cluster(text)
sentences, emb_sentences  = cluster.get_embedding_sentence()
summ_cluster = cluster.get_summary()
# cluster.evaluation_rouge(summ_cluster, summ)
