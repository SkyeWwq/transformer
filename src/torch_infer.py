# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 11:38 上午
# @Author  : Dawein
# @File    : torch_infer.py
# @Software : PyCharm

"""
推断
"""

import os
import torch
import numpy as np
from data_generator import Vocab
from torch_transformer import Transformer

class Infer:
    def __init__(self, pConfig):
        self.pConfig = pConfig
        self.vocab = Vocab(pConfig.vocab_file)
        self.token2idx = self.vocab.token2idx
        self.idx2token = self.vocab.idx2word
        self.src_max_len = pConfig.src_max_len
        self.trg_max_len = pConfig.trg_max_len
        self.PAD = self.token2idx[self.vocab.special_tokens["PAD"]]
        self.BOS = self.token2idx[self.vocab.special_tokens["BOS"]]
        self.EOS = self.token2idx[self.vocab.special_tokens["EOS"]]
        self.UNK = self.token2idx[self.vocab.special_tokens["UNK"]]

    def split_by_word(self, x):
        res = []
        x = x.strip()

        tmp = ""
        for ch in x:
            if '\u4e00' <= ch <= '\u9fff' or ch == " ":
                if tmp != "":
                    res.append(tmp)
                    tmp = ""
                res.append(ch)
            else:
                tmp += ch
        if tmp != "":
            res.append(tmp)
        return res


    def _word2idx(self, x):
        batch_size = len(x)
        source_x = np.zeros(shape=[batch_size, self.src_max_len], dtype=np.int) + self.PAD
        for i, s in enumerate(x):
            if isinstance(s, str):
                s = self.split_by_word(s)
            s = [self.token2idx[w] for w in s]
            s = s[:self.src_max_len - 1] + [self.EOS]
            source_x[i, :len(s)] = s[:]

        return source_x

    def _idx2word(self, y):
        rst = []
        for s in y:
            text = []
            for id in s:
                if id == self.EOS:
                    break
                text.append(self.idx2token[id])
            rst.append(text)
        return rst

    def infer(self):
        print("Start to infer......")
        beam_size = self.pConfig.beam_size
        enc_layer_num = self.pConfig.enc_layer_num
        dec_layer_num = self.pConfig.dec_layer_num
        n_heads = self.pConfig.n_heads
        d_model = self.pConfig.d_model
        d_ff = self.pConfig.d_ff
        dropout = self.pConfig.dropout
        src_max_len = self.pConfig.src_max_len
        trg_max_len = self.pConfig.trg_max_len
        src_vocab_size = self.vocab.size
        trg_vocab_size = self.vocab.size
        share_embeddings = self.pConfig.share_embeddings
        share_embeddings_proj = self.pConfig.share_embeddings_prob

        symbols = {"PAD": self.vocab.token2idx[self.pConfig.PAD],
                   "BOS": self.vocab.token2idx[self.pConfig.BOS],
                   "EOS": self.vocab.token2idx[self.pConfig.EOS],
                   "UNK": self.vocab.token2idx[self.pConfig.UNK]}

        # create model
        model = Transformer(src_vocab_size, src_max_len,
                            trg_vocab_size, trg_max_len,
                            n_heads, d_model, d_ff, enc_layer_num, dec_layer_num,
                            symbols, dropout, share_embeddings, share_embeddings_proj)

        if self.pConfig.use_cuda:
            model = model.cuda()

        # load model
        model_dir = self.pConfig.model_dir
        model_name = self.pConfig.model_name
        model_name = os.path.join(model_dir, model_name)
        print("load model from: {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

        while True:
            x = input("请输入: ")
            if x == "q":
                break
            x = [x]
            source_x = self._word2idx(x)
            if self.pConfig.use_cuda:
                source_x = torch.tensor(source_x, dtype=torch.long).cuda()
            else:
                source_x = torch.tensor(source_x, dtype=torch.long)

            hypothesis, log_probs = model.beam_search(source_x, beam_size, self.trg_max_len)
            hypothesis = hypothesis.detach().numpy()
            log_probs = log_probs.detach().numpy()
            log_probs = log_probs[:, -1]
            sentences = self._idx2word(hypothesis)
            rst = []
            for text, score in zip(sentences, log_probs):
                rst.append(["".join(text), score])
            rst.sort(key=lambda x: x[1], reverse=True)
            for text, score in rst:
                print("{}\t{}".format(text, score))