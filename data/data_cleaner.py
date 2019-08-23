# encoding=utf-8

import codecs

tf = 'han_qing_true.txt'
ff = 'han_qing_false.txt'

with codecs.open(tf, 'r', encoding='utf-8') as f, \
        codecs.open('sents.txt', 'w', encoding='utf-8') as f_out:

    for line in f:
        line = line.strip()
        if not (line.startswith('&& ') or line.endswith(' &&')):
            f_out.write(line + '=' + '0\n')

with codecs.open(ff, 'r', encoding='utf-8') as f, \
        codecs.open('sents.txt', 'a', encoding='utf-8') as f_out:

    for line in f:
        line = line.strip()
        if not (line.startswith('&& ') or line.endswith(' &&')):
            f_out.write(line + '=' + '1\n')
