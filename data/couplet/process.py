# -*- coding: utf-8 -*-
# @Time    : 2020/9/2 4:14 下午
# @Author  : Dawein
# @File    : process.py
# @Software : PyCharm

import random
from collections import defaultdict

couplet_in = []
with open("in.txt", encoding='utf-8') as fobj:
    for line in fobj:
        couplet_in.append(line.strip())

couplet_out = []
with open("out.txt", encoding='utf-8') as fobj:
    for line in fobj:
        couplet_out.append(line.strip())


couplets = []
for q, r in zip(couplet_in, couplet_out):
    couplets.append([q, r])


# 构建词表
vocabs = defaultdict(int)
for q, r in couplets:
    q = q.split()
    r = r.split()
    for token in q:
        vocabs[token] += 1
    for token in r:
        vocabs[token] += 1
vocabs = sorted(vocabs.items(), key=lambda x: x[1], reverse=True)
with open("vocab.txt", "w", encoding='utf-8') as fobj:
    for token, freq in vocabs:
        fobj.write("{}\t{}\n".format(token, freq))

samples = random.sample(couplets, int(len(couplets) * 0.1))
with open("train.txt", "w", encoding='utf-8') as fobj:
    for q, r in samples:
        fobj.write("{}\t{}\n".format(q, r))