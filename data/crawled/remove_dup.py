import codecs
from langconv import Converter

converter = Converter('zh-hans')
sents_set = set()

with codecs.open('crawled_dup.txt', 'r', encoding='utf-8') as f1, \
        codecs.open('crawled.txt', 'w', encoding='utf-8') as f2:
    for line in f1:
        line = converter.convert(line)
        if line not in sents_set:
            sents_set.add(line)
            f2.write(line)
