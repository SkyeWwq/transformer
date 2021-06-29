# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 10:23 上午
# @Author  : Dawein
# @File    : data_generator.py
# @Software : PyCharm

"""
数据生成器
"""

import random
import numpy as np

class Vocab:
    def __init__(self, vocab_file, add_special=True):
        self.special_tokens = {"PAD": "<PAD>",
                               "BOS": "<BOS>",
                               "EOS": "<EOS>",
                               "UNK": "<UNK>"}
        vocabs = []
        if add_special:
            for key in self.special_tokens:
                vocabs.append(self.special_tokens[key])

        with open(vocab_file, encoding="utf-8") as fobj:
            for line in fobj:
                lines = line.strip().split("\t")
                token = lines[0]
                vocabs.append(token)

        self.word2idx = {w:idx for idx, w in enumerate(vocabs)}
        self.idx2word = {idx:w for idx, w in enumerate(vocabs)}

    @property
    def token2idx(self):
        return self.word2idx

    @property
    def idx2token(self):
        return self.idx2word

    @property
    def size(self):
        return len(self.token2idx)

    @property
    def symbols(self):
        return self.special_tokens

class DateGenerator():

    def __init__(self, token2idx, special_tokens, src_max_len, trg_max_len):
        self.token2idx = token2idx
        self.PAD = special_tokens["PAD"]
        self.BOS = special_tokens["BOS"]
        self.EOS = special_tokens["EOS"]
        self.UNK = special_tokens["UNK"]

        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    ## 数据处理
    def _vectorize(self, data_in):
        batch_size = len(data_in)
        source_x = np.zeros(shape=[batch_size, self.src_max_len], dtype=np.int) + self.PAD
        source_y = np.zeros(shape=[batch_size, self.trg_max_len], dtype=np.int) + self.PAD
        target_y = np.zeros(shape=[batch_size, self.trg_max_len], dtype=np.int) + self.PAD

        x_lens = np.zeros(shape=[batch_size], dtype=np.int)
        y_lens = np.zeros(shape=[batch_size], dtype=np.int)

        for i, (x, y) in enumerate(data_in):
            source_x[i, :len(x)] = x[:]
            source_y[i, :len(y)-1] = y[:-1]
            target_y[i, :len(y)-1] = y[1:]

            x_lens[i] = len(x)
            y_lens[i] = len(y) - 1

        return source_x, source_y, target_y, x_lens, y_lens

    def batcher(self, data_file, batch_size, batch_num):

        data = []
        with open(data_file, encoding="utf-8") as fobj:
            for line in fobj:
                line = line.strip()
                x, y = line.split("\t")

                # token to id
                x = [self.token2idx.get(w, self.UNK) for w in x.split()]
                y = [self.token2idx.get(w, self.UNK) for w in y.split()]
                x = x[:self.src_max_len-1] + [self.EOS]
                y = [self.BOS] + y[:self.trg_max_len-1] + [self.EOS]
                data.append([x, y])

                if len(data) == batch_size * batch_num:
                    # random.shuffle(data)
                    batch_data = data[:batch_size]
                    yield self._vectorize(batch_data)
                    data = data[batch_size:]
                    break

            # process residual data
            while len(data):
                batch_data = data[:batch_size]
                yield self._vectorize(batch_data)
                data = data[batch_size:]

## main
if __name__ == '__main__':
    vocab_file = "../data/sen_func/vocab.txt"
    vocab = Vocab(vocab_file=vocab_file)

    print(vocab.token2idx)

    # data_file = "../data/sen_func/train.txt"
    # ops = DateGenerator(token2idx=vocab.token2idx, special_tokens=vocab.special_tokens,
    #                     src_max_len=30, trg_max_len=30)
    # for data in ops.batcher(data_file, 10, 2):
    #     print(data)