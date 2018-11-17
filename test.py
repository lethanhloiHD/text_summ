import re

s = " ngày 12 11 2018 "
s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s)

print(s)