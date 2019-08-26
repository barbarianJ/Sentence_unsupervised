import codecs
from langconv import Converter
from os import rename, remove

converter = Converter('zh-hans')
sents_set = set()

rename('crawled.txt', 'crawled_dup.txt')

with codecs.open('crawled_dup.txt', 'r', encoding='utf-8') as f1, \
        codecs.open('crawled.txt', 'w', encoding='utf-8') as f2:
    """
    convert to simplified chinese 
    and remove duplicated sentences in crawled.txt
    """
    for line in f1:
        line = converter.convert(line)
        if line not in sents_set:
            sents_set.add(line)
            f2.write(line)

remove('crawled_dup.txt')
