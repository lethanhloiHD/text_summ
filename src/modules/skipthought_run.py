from src.Graph_base import *
from src.modules.doc2vec import *
import nltk
import  numpy as np
import os
import time
from skip_thoughts.evaluate import *
from skip_thoughts.train import *


# load_data_train_skipthought(directory_train)
# train_skipthought()

# print("===========time train skipthought : {}s===========".format(time.time()-s))
usable_encoder = UsableEncoder()
X = [
    'Tối 10/12, sau một ngày bủa vây, gây sức ép và gây rối tại trụ sở VFF để đòi mua vé xem trận chung kết lượt về AFF Cup giữa đội tuyển Việt Nam - Malaysia, một số người xưng thương binh đã tụ tập ăn nhậu ngay tại trụ sở VFF.',
    "Vị luật sư này nhấn mạnh, trong trường hợp, sau khi xác minh, có những người không phải thương binh mà giả danh để trà trộn vào kích động, gây rối cần xử lý nghiêm minh theo quy định của pháp luật",
    "Bên cạnh đó, luật sư Cường cho rằng, ngoài việc lên án, xử lý nghiêm những người có hành vi gây rối cũng cần xem xét trách nhiệm của Liên đoàn Bóng đá Việt Nam (VFF), bởi trong việc tổ chức bán vé làm chưa tốt, chưa hết trách nhiệm nên mới dẫn đến hệ quả này.",
    "Thực tế, không phải bây giờ mới có những trường hợp gây rối như trên ở cổng, trụ sở VFF mà trước đây cũng xảy ra nhiều nhưng đơn vị này lại không có biện pháp phòng ngừa, xử lý hữu hiệu, giải quyết triệt để",
    "Ngoài ra, dư luận cũng lên tiếng nhiều về vấn đề chưa tốt trong khâu bán vé của Liên đoàn trong thời gian qua",
    "Do đó, cần có sự phán xét, đánh giá trách nhiệm của cả hai bên người hâm mộ, nhóm người xưng thương binh kia và Liên đoàn Bóng đá Việt Nam",
    "Khi mọi việc công khai, minh bạch sẽ tránh đi những hành vi không đẹp"

]
x_vec = (usable_encoder.encode(X))
print(x_vec[0])

# y = [
#     'Thái lan',
# ]
# y_vec = (usable_encoder.encode(y).reshape(1200,))
# print(y_vec)
# print(cosine_similarity_vector(x_vec,y_vec))

# sen2index = {}
# sents = split_sentences(text)
# for index, sent in enumerate(sents):
#     sen2index.update({
#         sent: index
#     })
#     print(index,sent)


# a = score_position_sentence(9,0.2,9)
# print(a)
