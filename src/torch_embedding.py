# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 10:29 上午
# @Author  : Dawein
# @File    : torch_embedding.py
# @Software : PyCharm

"""
embedding
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedds = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = self.embedds[input]
        return output

class PositionEmbedding(nn.Module):
    def __init__(self, seq_maxlen, d_model, is_training=False):
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        self.is_training = is_training
        # position embeddings, if no need train, calc based on paper.
        if not self.is_training:
            embedds = torch.zeros(seq_maxlen, self.d_model)
            for i in range(seq_maxlen):
                for j in range(0, self.d_model, 2):
                    embedds[i, j] = math.sin(i / math.pow(10000, 2 * j / self.d_model))
                    embedds[i, j + 1] = math.cos(i / math.pow(10000, 2 * j / self.d_model))
            self.register_buffer("embedds", embedds)
        else:
            self.embedds = nn.Parameter(torch.Tensor(seq_maxlen, self.d_model))
            self.init_parameters()

    def init_parameters(self):

        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, input, start_pos=0):
        seq_len = input.size(1)
        if not self.is_training:
            pos_embedds = self.embedds[start_pos:start_pos+seq_len, :]
            pos_embedds = Variable(pos_embedds, requires_grad=False)
        else:
            pos_embedds = self.embedds[start_pos:start_pos+seq_len, :]

        return pos_embedds
