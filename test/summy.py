from sumy.parsers.plaintext import PlaintextParser  # We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer  # We're choosing Lexrank, other algorithms are also built in
import rouge

file = "data/text/4c1b1fee449ea5c30205832c8e4a632a_body.txt"  # name of the plain-text file

parser = PlaintextParser.from_file(file, Tokenizer("english"))
summarizer = LexRankSummarizer()

summary = summarizer(parser.document, 5)  # Summarize the document with 5 sentences

result = ''
for sentence in summary:
    print(sentence)
    result += str(sentence) +" \n"


sum = """ Thành Cát Tư Hãn , họ Bột Nhi Chỉ Cân tên đầy đủ là Bột Nhi Chỉ Cân Thiết Mộc Chân sinh vào năm 1162 và mất " \
          "ngày 18 tháng 8 năm 1227, là Hãn vương của Mông Cổ và là người sáng lập ra Đế quốc Mông Cổ " \
          "sau khi hợp nhất các bộ lạc độc lập ở vùng đông bắc châu Á năm 1206. Là một nhà lãnh đạo lỗi lạc và quan trọng của lịch sử " \
          "thế giới, ông được người Mông Cổ dành cho sự tôn trọng cao nhất, như là một vị lãnh đạo đã loại bỏ hàng thế kỷ của các " \
          "cuộc giao tranh, mang lại sự ổn định về chính trị và kinh tế cho khu vực Á-Âu trong lãnh thổ của ông, mặc dù đã gây ra " \
          "những tổn thất to lớn đối với những người chống lại ông. Cháu nội của ông và là người kế tục sau này, đại hãn Hốt Tất Liệt " \
          "đã thiết lập ra triều đại nhà Nguyên của Trung Quốc. Tháng 10 năm Chí Nguyên thứ 3 (1266), Hốt Tất Liệt đã truy tôn " \
          "Thành Cát Tư Hãn miếu hiệu là Thái Tổ, nên ông còn được gọi là Nguyên Thái Tổ. Thụy hiệu khi đó truy tôn là " \
          "Thánh Vũ Hoàng đế. Tới năm Chí Đại thứ 2 (1309), Nguyên Vũ Tông Hải Sơn gia thụy thành Pháp Thiên Khải Vận." \
          " Từ đó thụy hiệu của ông trở thành Pháp Thiên Khải Vận Thánh Vũ Hoàng đế. " \
          "Có rất nhiều nhân vật nổi tiếng được cho là hậu duệ của Thành Cát Tư Hãn, là những kẻ đi xâm chiếm nhiều đất đai về " \
          "tay mình như Timur Lenk, kẻ chinh phục dân Thổ Nhĩ Kỳ, Babur, người sáng lập ra đế quốc Mogul trong lịch sử Ấn Độ. " \
          "Những hậu duệ khác của Thành Cát Tư Hãn còn tiếp tục cai trị Mông Cổ đến thế kỷ 17 cho đến khi nó bị người Trung Quốc " \
          "thống trị lại."""


evaluator = rouge.Rouge(metrics=['rouge-n'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            # alpha=0.5, # Default F1_score
                            # weight_factor=1.2,
                            stemming=True)

scores = evaluator.get_scores(result, sum)
print(scores)
