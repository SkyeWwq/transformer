# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 10:20 上午
# @Author  : Dawein
# @File    : torch_dropout.py
# @Software : PyCharm

"""
dropout
"""

import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout ratio must be in [0, 1].")
        self.p = p

    def forward(self, input):
        if self.training:
            binary = (torch.rand_like(input, device=input.device) > self.p).float()
            if self.p == 1:
                scale = 1
            else:
                scale = 1 / (1 - self.p)
            input = input * binary * scale
        return input
