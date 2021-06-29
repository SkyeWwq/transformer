# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 11:31 上午
# @Author  : Dawein
# @File    : torch_feedforward.py
# @Software : PyCharm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dropout import Dropout

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p=0):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = Dropout(drop_p)

        self.W0 = nn.Parameter(torch.Tensor(d_ff, d_model))
        self.b0 = nn.Parameter(torch.Tensor(d_ff))
        self.W1 = nn.Parameter(torch.Tensor(d_model, d_ff))
        self.b1 = nn.Parameter(torch.Tensor(d_model))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.d_model)
        for weight in [self.W0, self.W1]:
            weight.data.uniform_(-stdv, stdv)
        for weight in [self.b0, self.b1]:
            weight.data.fill_(0)

    def forward(self, input):
        output = F.linear(input, self.W0, self.b0)
        output = F.linear(F.relu(output), self.W1, self.b1)

        output = self.dropout(output)
        return output