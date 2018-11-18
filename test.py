import re
from config import *

from nltk.tokenize import RegexpTokenizer


s = "tp. hcm mr. hang nam 2019 rat dep"
tokenizer = RegexpTokenizer(r'\w+')
s_ = tokenizer.tokenize(s)
print(s_)
# for i,v in mapping.items():
#     if i in s :
#         s = s.replace(i,v)
#
# print(s)