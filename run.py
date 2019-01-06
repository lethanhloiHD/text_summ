from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
from pyvi import ViTokenizer,ViPosTagger
import  numpy as np
# from skip_thoughts.evaluate import *
# from skip_thoughts.train import *

# data_test= load_data_test()
# avg_rouge(data_test)

# text ="""Bỏ lại đứa con gái 14 tháng tuổi ngay tại công viên, người mẹ cầu xin mọi người nuôi con giúp
# vì đang gặp khó khăn. Nhìn đứa trẻ khóc thét,
# thẫn thờ tìm mẹ khi trong người đang mắc phải bệnh tay - chân - miệng khiến ai cũng xúc động. Theo thông tin ban đầu cho biết, khoảng 19h ngày 4/1, người dân sinh sống gần công viên thị trấn Tân Phú,
# huyện Đồng Phú, tỉnh Bình Phước phát hiện một bé gái bị mẹ bỏ rơi,
# òa khóc nức nở đi tìm mẹ ngay tại công viên. Trên người bé gái có một túi quần áo kèm theo lá thư do chính người mẹ để lại với nội dung: "Tôi là mẹ không có điều kiện nuôi con. Vậy ai nhặt được cháu nhờ nuôi dưỡng chăm sóc cháu, cháu sinh ngày 12/10/2017. Cháu chưa làm giấy khai sinh".
# Dù được người dân dỗ dành nhưng bé gái vẫn liên tục òa khóc nức nở, ánh mắt thẫn thờ tìm mẹ. "Bé gái hiện bị bệnh tay chân miệng, giờ lại không có mẹ ở bên, bé cứ khóc liên tục đòi mẹ. Không biết người nhà của bé gái ở đâu, vì bé gái cũng lớn rồi, ai biết được xin giúp bé gái được về với mẹ. Bé cần tình thương,
# hơi ấm của mẹ chứ không chịu nỗi cảnh xa mẹ đâu", một người dân cho biết. Sau khi phát hiện bé gái bị bỏ rơi, người dân đã trình báo lên chính quyền địa phương. Đồng thời, bé cũng được đưa vào bệnh viện Đa khoa huyện Đồng Phú để nhập viện điều trị bệnh tay - chân - miệng cấp độ 2.
# Nếu ai biết được thông tin gia đình bé xin vui lòng thông báo giúp, đến bệnh viện huyện để gặp con. Bé gái thật sự đang cần gặp mẹ và được sự chăm sóc của mẹ hơn bất cứ ai khác.
# """
# #
# # summ = """Trong các bữa tiệc của hai nhà hàng (thuộc khách sạn Sofitel Legend Metropole Hà Nội), cơm và thịt bò là thứ thường bị bỏ lại nhiều nhất. Đó là kết quả sau một năm rưỡi áp dụng công nghệ Winnow giúp các đầu bếp đo lường, kiểm tra và cắt giảm thực phẩm dư thừa. Các đầu bếp sử dụng kết quả phân tích để cắt giảm thực phẩm hay bị thừa, hạn chế việc chế biến quá nhiều thức ăn. Sau một năm rưỡi, hai nhà hàng trên tiết kiệm được 11,2 tấn thức ăn, giúp tiết kiệm hơn 1 tỷ đồng. Hiện 38 khách sạn thuộc AccorHotels trong khu vực, trong đó có 6 khách sạn tại Việt Nam đã cắt giảm lượng lớn chất thải thực phẩm nhờ áp dụng công nghệ này."""
# # data_test= load_data_test()
# # text = data_test[0]['text']
# # summ = data_test[0]['summ']
# lexrank = Rank(text,tfidf_option=False,
#                     doc2vec_option=False,
#                     word2vec_option=False,
#                     autoencoder_option=False
#             )
# sum_lexrank = lexrank.summary(option_mmr=True, using_postion_score= True)

# print(lexrank.evaluation_rouge(sum_lexrank,summ))

import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-f','--foo', help='Description for foo argument', required=True)
parser.add_argument('-b','--bar', help='Description for bar argument', required=True)
args = (parser.parse_args())
if args.foo == 'tf'and args.bar == 'graph':
    print("viet nam vo dich")




