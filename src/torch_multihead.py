# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 2:47 下午
# @Author  : Dawein
# @File    : torch_multihead.py
# @Software : PyCharm

"""
Transformer: multi-head attention
Multi-head(Q, K, V) = softmax(QK') / sqrt(d) * V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# scale-dot-product attention
def scaled_dot_product_attention(query, key, value, attn_mask=None):

    d = query.size(-1)
    # B * n_heads * L_q * L_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)

    if attn_mask is not None:
        attn_mask = attn_mask.type(torch.bool)
        scores = scores.masked_fill(attn_mask, -1e-7)

    attn_scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, value)

    return output, attn_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout_r=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d = self.d_model // self.n_heads
        # dropout
        self.W_q = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
        self.W_k = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
        self.W_v = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
        self.W_o = nn.Parameter(torch.Tensor(self.d_model, self.d_model))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.d_model)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, query, key, value, attn_mask=None):

        # B * L * d_model -> B * L * (n_heads * d)
        query = F.linear(query, self.W_q)
        key = F.linear(key, self.W_k)
        value = F.linear(value, self.W_v)

        # split: B * L * n_heads * d
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.n_heads, self.d)
        key = key.view(batch_size, -1, self.n_heads, self.d)
        value = value.view(batch_size, -1, self.n_heads, self.d)

        # transpose: B * n_head * L * d
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        concat, attn_scores = scaled_dot_product_attention(query, key, value, attn_mask)

        # B * n_heads * L * d -> B * L * n_heads * d -> B * L * d_model
        concat = concat.transpose(1, 2)
        concat = concat.contiguous().view(batch_size, -1, self.d_model)
        output = F.linear(concat, self.W_o)
        return output, attn_scores