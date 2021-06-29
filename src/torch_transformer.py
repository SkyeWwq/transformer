# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 8:20 下午
# @Author  : Dawein
# @File    : torch_transformer.py
# @Software : PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_encoder import Encoder
from torch_decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, trg_vocab_size, trg_max_len,
                 n_heads, d_model, d_ff, enc_layer_num, dec_layer_num, special_tokens,
                 dropout=0, share_embeddings=False, share_embed_out_proj=False):
        super(Transformer, self).__init__()
        self.PAD = special_tokens["PAD"] # id
        self.BOS = special_tokens["BOS"]
        self.EOS = special_tokens["EOS"]
        self.UNK = special_tokens["UNK"]

        self.trg_vocab_size = trg_vocab_size
        self.dec_layer_num = dec_layer_num
        self.d_model = d_model
        self.trg_max_len = trg_max_len

        # encoder
        self.encoder = Encoder(src_vocab_size, src_max_len, n_heads, d_model,
                               d_ff, enc_layer_num, dropout)

        # share embeddings
        embeddings = None
        if share_embeddings:
            embeddings = self.encoder.token_embedds

        # decoder
        self.decoder = Decoder(trg_vocab_size, trg_max_len, n_heads, d_model,
                               d_ff, dec_layer_num, dropout, embeddings, share_embed_out_proj)

    # get attn mask
    def get_attn_mask(self, query, key):

        seq_len = query.size(1)
        mask = key.eq(self.PAD)
        mask = mask.unsqueeze(1).repeat(1, seq_len, 1).type(torch.uint8)
        return mask

    # get directional mask
    def get_directional_mask(self, input):
        seq_len = input.size(1)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=input.device, dtype=torch.uint8),
                          diagonal=1)
        return mask

    # forward
    def forward(self, source, target):

        # create mask
        enc_self_attn_mask = self.get_attn_mask(source, source)
        dec_self_attn_mask = self.get_attn_mask(target, target)

        # during training stage, the previous token can not see the later token.
        dec_directional_mask = self.get_directional_mask(target)
        dec_self_attn_mask = dec_self_attn_mask | dec_directional_mask

        dec_enc_attn_mask = self.get_attn_mask(target, source)

        # source encoder
        enc_output, _ = self.encoder(source, enc_self_attn_mask)

        # target deocder
        dec_output = self.decoder(target, enc_output, dec_self_attn_mask, dec_enc_attn_mask)

        # return
        return dec_output

    # inference stage
    def beam_step(self, input_y, steps, enc_kv_list, dec_enc_attn_mask,
                  dec_kv_list, last_log_probs, mask_finished, finished,
                  batch_size, beam_size, vocab_size, expand, hypothesis):

        dec_kv_list, logits, _, _ = self.decoder.step(input_y, steps, enc_kv_list,
                                                      dec_enc_attn_mask, dec_kv_list)
        mask = finished.type(torch.float)
        log_probs = F.log_softmax(logits, 1) * (1 - mask)
        log_probs = log_probs + mask_finished * mask
        log_probs = last_log_probs[:, -1:] + log_probs

        if expand:
            log_probs, indice = log_probs.topk(beam_size)
            log_probs = log_probs.view(-1, 1)
            beam_id = torch.arange(0, batch_size).unsqueeze(1)
            beam_id = beam_id.repeat(1, beam_size).view(-1)
            input_y = indice.view(-1, 1)

            for i in range(self.dec_layer_num):
                enc_kv_list[i][0] = enc_kv_list[i][0][beam_id]
                enc_kv_list[i][1] = enc_kv_list[i][1][beam_id]
                dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
                dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]
            dec_enc_attn_mask = dec_enc_attn_mask[beam_id]
        else:
            log_probs = log_probs.view(-1, beam_size * vocab_size)
            log_probs, indice = log_probs.topk(beam_size)
            log_probs = log_probs.view(-1, 1)
            offset = torch.arange(0, batch_size, device=input_y.device) * beam_size
            beam_id = (offset.view(-1, 1) + indice // vocab_size).view(-1)
            input_y = (indice % vocab_size).view(-1, 1)
            for i in range(self.dec_layer_num):
                dec_kv_list[i][0] = dec_kv_list[i][0][beam_id]
                dec_kv_list[i][1] = dec_kv_list[i][1][beam_id]

        log_probs = log_probs - finished[beam_id].type(torch.float) * 10000
        log_probs = torch.cat([last_log_probs[beam_id, :], log_probs], 1)
        finished = (finished[beam_id, :] | input_y.eq(self.EOS).type(finished.dtype))

        hypothesis = torch.cat([hypothesis[beam_id, :], input_y.view(-1, 1)], 1)

        return input_y, enc_kv_list, dec_enc_attn_mask, \
               dec_kv_list, log_probs, finished, hypothesis

    def init_beam_search(self, batch_size, beam_size, use_cuda=False):

        vocab_size = self.trg_vocab_size
        dev_kv_list = []
        for i in range(self.dec_layer_num):
            key = torch.zeros(batch_size, 0, self.d_model)
            value = torch.zeros(batch_size, 0, self.d_model)
            if use_cuda:
                key = key.cuda()
                value = value.cuda()
            dev_kv_list.append([key, value])

        init_y = torch.ones(batch_size, 1, dtype=torch.long)
        log_probs = torch.zeros(batch_size, 1, dtype=torch.float)
        finished = torch.zeros(batch_size, 1, dtype=torch.uint8)
        hypothesis = torch.ones(batch_size*beam_size, 0, dtype=torch.long)

        mask_finished = torch.tensor([-10000] * vocab_size, dtype=torch.float)
        mask_finished[self.EOS] = 10000

        if use_cuda:
            init_y = init_y.cuda()
            log_probs = log_probs.cuda()
            finished = finished.cuda()
            hypothesis = hypothesis.cuda()
            mask_finished = mask_finished.cuda()

        return init_y, dev_kv_list, log_probs, finished, hypothesis, mask_finished

    ## beam search
    def beam_search(self, input_x, beam_size, max_dec_steps):

        use_cuda = input_x.is_cuda
        enc_self_attn_mask = self.get_attn_mask(input_x, input_x)
        enc_output, _ = self.encoder(input_x, enc_self_attn_mask)
        enc_kv_list = self.decoder.cache_enc_kv(enc_output)

        batch_size = input_x.size(0)
        input_y, dev_kv_list, log_probs, finished, hypothesis, mask_finished \
            = self.init_beam_search(batch_size, beam_size, use_cuda)

        dec_enc_attn_mask = self.get_attn_mask(input_y, input_x)
        vocab_size = self.trg_vocab_size
        steps = 0
        max_dec_steps = min(self.trg_max_len, max_dec_steps)
        while not finished.all() and steps < max_dec_steps:
            if steps == 0:
                expand = True
            else:
                expand = False

            input_y, enc_kv_list, dec_enc_attn_mask, \
            dec_kv_list, log_probs, finished, hypothesis \
                = self.beam_step(input_y, steps, enc_kv_list, dec_enc_attn_mask,
                                 dev_kv_list, log_probs, mask_finished, finished,
                                 batch_size, beam_size, vocab_size, expand, hypothesis)
            steps = steps + 1

        return hypothesis, log_probs