# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 1:50 下午
# @Author  : Dawein
# @File    : torch_layerNorm.py
# @Software : PyCharm

"""
layer normalization
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))


    def forward(self, input:torch.Tensor)->torch.Tensor:
        eps = 1e-7
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True, unbiased=False)
        norm = self.alpha * (input - mean) / (std + eps) + self.beta
        return norm