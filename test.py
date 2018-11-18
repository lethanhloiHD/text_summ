import re
from config import *


s = "tp. hcm mr. hang nam 2019 rat dep"
for i,v in mapping.items():
    if i in s :
        s = s.replace(i,v)

print(s)