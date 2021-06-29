# -*- coding: utf-8 -*-
# @Time    : 2020/8/4 3:02 下午
# @Author  : Dawein
# @File    : torch_encoder.py
# @Software : PyCharm

"""
Encoder: self-attention
"""

import torch
import torch.nn as nn
from torch_dropout import Dropout
from torch_embedding import Embedding, PositionEmbedding
from torch_attention import MultiHeadAttention
from torch_feedforward import FeedForward
from torch_layerNorm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_p=0):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p = dropout_p

        self.self_attention = MultiHeadAttention(self.n_heads, self.d_model, self.p)
        self.fnn = FeedForward(self.d_model, self.d_ff, self.p)
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)

    def forward(self, enc_output, attn_mask):
        # residual
        residual = enc_output
        enc_output, attn_scores = self.self_attention(enc_output,
                                                      enc_output,
                                                      enc_output,
                                                      attn_mask)
        enc_output = self.norm1(residual + enc_output)

        residual = enc_output
        enc_output = self.fnn(enc_output)
        enc_output = self.norm2(residual + enc_output)

        return enc_output, attn_scores

class Encoder(nn.Module):

    def __init__(self, vocab_size, max_len, n_heads, d_model, d_ff, n_layers, dropout=0):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = Dropout(dropout)
        self.token_embedds = Embedding(vocab_size, d_model)
        self.position_embedds = PositionEmbedding(max_len, d_model, is_training=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout)
                                     for _ in range(self.n_layers)])


    def forward(self, input, attn_mask):

        enc_self_attn_list = []
        enc_output = self.token_embedds(input) + self.position_embedds(input)
        enc_output = self.dropout(enc_output)
        for enc_layer in self.layers:
            enc_output, enc_attn_scores = enc_layer(enc_output, attn_mask)
            enc_self_attn_list.append(enc_attn_scores)

        return enc_output, enc_self_attn_list

