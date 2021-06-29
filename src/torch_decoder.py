# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 7:17 下午
# @Author  : Dawein
# @File    : torch_decoder.py
# @Software : PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_embedding import Embedding, PositionEmbedding
from torch_attention import MultiHeadAttention
from torch_layerNorm import LayerNorm
from torch_feedforward import FeedForward
from torch_dropout import Dropout

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout=0):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.enc_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.fnn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, enc_output, dec_output, self_attn_mask, dec_enc_attn_mask):

        # decoder self-attention
        residual = dec_output
        dec_output, self_attn_scores = self.self_attention(dec_output, dec_output,
                                                           dec_output, self_attn_mask)

        # add and normal
        dec_output = self.norm1(residual + dec_output)

        # encoder-decoder attention
        residual = dec_output
        dec_output, dec_enc_attn_scores = self.enc_attention(dec_output, enc_output,
                                                             enc_output, dec_enc_attn_mask)

        # add and normal: LayerNorm(x + Sublayer(x))
        dec_output = self.norm2(residual + dec_output)

        # feed forward + add & layer normalization
        residual = dec_output
        dec_output = self.fnn(dec_output)
        dec_output = self.norm3(residual + dec_output)

        # return
        return dec_output, self_attn_scores, dec_enc_attn_scores

    # cache
    def cache_enc_kv(self, enc_output):

        enc_output = self.enc_attention.cache_kv(enc_output)
        return enc_output

    def cache_dec_kv(self, dec_output):

        dec_output = self.self_attention.cache_kv(dec_output)
        return dec_output

    def forward_with_cache(self, enc_keys, enc_values, dec_enc_attn_mask,
                           dec_keys, dec_values, dec_output):

        residual = dec_output
        dec_output, self_attn_scores = self.self_attention.forward_with_cache(dec_output,
                                                                              dec_keys,
                                                                              dec_values)

        dec_output = self.norm1(residual + dec_output)

        residual = dec_output
        dec_output, dec_enc_attn_scores = self.enc_attention.forward_with_cache(dec_output,
                                                                                enc_keys,
                                                                                enc_values,
                                                                                dec_enc_attn_mask)

        dec_output = self.norm2(residual + dec_output)

        residual = dec_output
        dec_output = self.fnn(dec_output)
        dec_output = self.norm3(residual + dec_output)

        return dec_output, self_attn_scores, dec_enc_attn_scores

## decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_heads, d_model,
                 d_ff, n_layers, dropout, embeddings, share_embed_out_proj=False):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = Dropout(dropout)

        if embeddings is not None:
            self.embeddigs = embeddings
        else:
            self.embeddigs = Embedding(vocab_size, d_model)

        self.pos_embeddings = PositionEmbedding(max_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout)
                                     for _ in range(self.n_layers)])

        self.share_embed_out_proj = share_embed_out_proj
        if self.share_embed_out_proj:
            self.W = self.embeddigs.embedds
        else:
            self.W = nn.Parameter(torch.Tensor(vocab_size, d_model))

        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / np.sqrt(self.d_model)
        if not self.share_embed_out_proj:
            self.W.data.uniform_(-stdv, stdv)

    def forward(self, input, enc_output, self_attn_mask, dec_enc_attn_mask):

        # embedding
        dec_output = self.embeddigs(input)
        dec_output = dec_output + self.pos_embeddings(input)
        dec_output = self.dropout(dec_output)

        for i, layer in enumerate(self.layers):
            dec_output, self_scores, dec_enc_scores = layer(enc_output, dec_output,
                                                            self_attn_mask, dec_enc_attn_mask)

        # linear
        dec_output = F.linear(dec_output, self.W)

        # logits
        logits = dec_output.view(-1, self.vocab_size)

        return logits

    def cache_enc_kv(self, enc_output):
        kv_list = []
        for dec_layer in self.layers:
            kv_list.append(dec_layer.cache_enc_kv(enc_output))

        return kv_list

    def cache_dec_kv(self, dec_output):
        kv_list = []
        for dec_layer in self.layers:
            kv_list.append(dec_layer.cache_dec_kv(dec_output))
        return kv_list

    # for inference stage decoding by step and step
    def step(self, input, steps, enc_kv_list, dec_enc_attn_mask, dec_kv_list):

        # embedding
        dec_output = self.embeddigs(input)
        dec_output = dec_output + self.pos_embeddings(input, steps)

        dec_self_attn_list = []
        dec_enc_attn_list = []
        for i, layer in enumerate(self.layers):
            kv = layer.cache_dec_kv(dec_output)
            dec_kv_list[i][0] = torch.cat([dec_kv_list[i][0], kv[0]], 1)
            dec_kv_list[i][1] = torch.cat([dec_kv_list[i][1], kv[1]], 1)
            dec_output, self_scores, dec_enc_scores = layer.forward_with_cache(
                enc_kv_list[i][0], enc_kv_list[i][1],
                dec_enc_attn_mask,
                dec_kv_list[i][0], dec_kv_list[i][1],
                dec_output)
            dec_self_attn_list.append(self_scores)
            dec_enc_attn_list.append(dec_enc_scores)

        dec_output = F.linear(dec_output, self.W)
        logits = dec_output.view(-1, self.vocab_size)

        return dec_kv_list, logits, dec_self_attn_list, dec_enc_attn_list