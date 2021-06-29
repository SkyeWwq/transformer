# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 9:24 上午
# @Author  : Dawein
# @File    : process.py
# @Software : PyCharm


"""
数据处理脚本
"""

import random
from collections import defaultdict

def check_chinese(x):
    for c in x:
        if not "\u4e00" <= c <= "\u9fa5":
            return False
    return True

def merge_post_response(post_file, response_file, file_out, vocab_file):

    def split_by_word(x):
        if isinstance(x, str):
            x = x.split(" ")

        out = []
        for token in x:
            if check_chinese(token):
                out += list(token)
            else:
                out.append(token)
        return out


    post = []
    with open(post_file, encoding="utf-8") as fobj:
        for line in fobj:
            line = line.strip()
            post.append(line)

    response = []
    with open(response_file, encoding="utf-8") as fobj:
        for line in fobj:
            line = line.strip()
            response.append(line)


    rst = []
    for x, y in zip(post, response):
        x = split_by_word(x)
        y = split_by_word(y)
        line = "{}\t{}\n".format(" ".join(x), " ".join(y))
        rst.append(line)

    # 构建词表
    vocab = defaultdict(int)
    for line in rst:
        lines = line.strip().split("\t")
        text = lines[0].split(" ") + lines[1].split(" ")
        for t in text:
            vocab[t] += 1

    # vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    # with open(vocab_file, "w", encoding="utf-8") as fobj:
    #     for token, freq in vocab:
    #         if freq < 100:
    #             continue
    #
    #         fobj.write("{}\t{}\n".format(token, freq))


    # 采样一部分数据
    samples = random.sample(rst, k=1000)

    with open(file_out, "w", encoding="utf-8") as fobj:
        for line in samples:
            fobj.write(line)

## main
if __name__ == '__main__':
    ## 合并post和response，并按照字级别分割
    p_file = "sen_func/DialogwithSenFun/weibo_pair_train_pattern.post"
    r_file = "sen_func/DialogwithSenFun/weibo_pair_train_pattern.response"
    f_out = "sen_func/train.txt"
    v_file = "sen_func/vocab.txt"
    merge_post_response(p_file, r_file, f_out, v_file)

