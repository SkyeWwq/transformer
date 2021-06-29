# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 2:17 下午
# @Author  : Dawein
# @File    : tf_transformer.py
# @Software : PyCharm

import numpy as np
import tensorflow as tf

## token embedding
class Embedding():

    def __init__(self, vocab_size, embedding_size, name="token_embedding"):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        with tf.variable_scope(name):
            self.embedds = tf.get_variable(name="embedds",
                                           shape=[self.vocab_size, self.embedding_size],
                                           initializer=tf.uniform_unit_scaling_initializer(-1/np.sqrt(embedding_size),
                                                                                           1/np.sqrt(embedding_size)),
                                           dtype=tf.float32)

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.embedds, x)

## position embedding
class PositionEmbedding():

    def __init__(self, max_seq_len, d_model, is_training=False, name="position_embedding"):
        self.max_len = max_seq_len
        self.d_model = d_model
        self.is_training = is_training

        if self.is_training:
            with tf.variable_scope(name):
                self.embedds = tf.get_variable(name="embedds",
                                               shape=[self.max_len, self.d_model],
                                               initializer=tf.uniform_unit_scaling_initializer(-1/np.sqrt(d_model),
                                                                                               1/np.sqrt(d_model)),
                                               dtype=tf.float32)
        else:
            # sin or cos
            self.embedds = np.zeros(shape=[self.max_len, self.d_model])
            for i in range(self.max_len):
                for j in range(self.d_model):
                    self.embedds[i, j] = np.sin(i / np.power(10000, 2 * j / d_model))
                    self.embedds[i, j + 1] = np.cos(i / np.power(10000, 2 * j / d_model))
            self.embedds = tf.constant(self.embedds, dtype=tf.float32)

    def __call__(self, x, start_pos=0):
        # B * L
        x_len = tf.shape(x)[1]
        return self.embedds[start_pos:start_pos+x_len, :]

# layer normalization
class LayerNorm():

    def __init__(self, d_model, name="layer_norm"):
        self.d_model = d_model
        with tf.variable_scope(name):
            self.alpha = tf.get_variable(name="alpha",
                                         shape=[self.d_model],
                                         initializer=tf.ones_initializer(),
                                         dtype=tf.float32)
            self.beta = tf.get_variable(name="beta",
                                        shape=[self.d_model],
                                        initializer=tf.ones_initializer(),
                                        dtype=tf.float32)

    def __call__(self, x):
        # calc mean and std
        eps = 1e-6
        mean, var = tf.nn.moments(x, [-1], keepdims=True)
        std = tf.sqrt(var)
        norm = self.alpha * (x - mean) / (std + eps) + self.beta
        return norm

## feed forward
class FeedForward():

    def __init__(self, d_model, d_ff, dropout, name="feed_forward"):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        init_parameters = tf.uniform_unit_scaling_initializer(-1/np.sqrt(self.d_model),
                                                              1/np.sqrt(self.d_model))
        with tf.variable_scope(name):
            self.W1 = tf.get_variable(name="W1",
                                      shape=[self.d_model, self.d_ff],
                                      initializer=init_parameters,
                                      dtype=tf.float32)
            self.b1 = tf.get_variable(name="b1",
                                      shape=[self.d_ff],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
            self.W2 = tf.get_variable(name="W2",
                                      shape=[self.d_ff, self.d_model],
                                      initializer=init_parameters,
                                      dtype=tf.float32)
            self.b2 = tf.get_variable(name="b2",
                                      shape=[self.d_model],
                                      initializer=init_parameters,
                                      dtype=tf.float32)

    def __call__(self, x):

        output = tf.matmul(self.W1, x) + self.b1
        output = tf.nn.relu(output)
        output = tf.matmul(self.W2, output) + self.b2
        output = tf.nn.dropout(output, rate=self.dropout)
        return output

# scalar_dot_product attention
class MultiHeadAttention():

    def __init__(self, n_heads, d_model, dropout=0, name="multi_head_attention"):
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.d = self.d_model // self.n_heads

        init_parameters = tf.uniform_unit_scaling_initializer(1/np.sqrt(d_model),
                                                              -1/np.sqrt(d_model))
        with tf.variable_scope(name):
            self.W_q = tf.get_variable(name="W_q",
                                       shape=[d_model, d_model],
                                       initializer=init_parameters,
                                       dtype=tf.float32)
            self.W_k = tf.get_variable(name="W_k",
                                       shape=[d_model, d_model],
                                       initializer=init_parameters,
                                       dtype=tf.float32)
            self.W_v = tf.get_variable(name="W_v",
                                       shape=[d_model, d_model],
                                       initializer=init_parameters,
                                       dtype=tf.float32)
            self.W_o = tf.get_variable(name="W_o",
                                       shape=[d_model, d_model],
                                       initializer=init_parameters,
                                       dtype=tf.float32)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None):

        float_min = -10000
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)

        scores = tf.matmul(query, tf.transpose(key, (0,1,3,2))) / tf.sqrt(d_k)

        if attn_mask is not None:
            scores = scores * attn_mask + float_min * (1 - attn_mask)

        attn_scores = tf.nn.softmax(scores, axis=-1)

        output = attn_scores * value
        return output, attn_scores

    def __call__(self, query, key, value, attn_mask=None):

        batch_size = tf.shape(query)[0]

        query = tf.tensordot(query, self.W_q, [2, 0])
        key = tf.tensordot(key, self.W_k, [2, 0])
        value = tf.tensordot(value, self.W_v, [2, 0])

        # B*L*d_model -> B*L*n_heads*d
        query = tf.reshape(query, shape=(batch_size, -1, self.n_heads, self.d))
        key = tf.reshape(key, shape=(batch_size, -1, self.n_heads, self.d))
        value = tf.reshape(value, shape=(batch_size, -1, self.n_heads, self.d))

        # B*L*n_head*d -> B*n_heads*L*d
        query = tf.transpose(query, (0, 2, 1, 3))
        key = tf.transpose(key, (0, 2, 1, 3))
        value = tf.transpose(value, (0, 2, 1, 3))

        if attn_mask is not None:
            attn_mask = tf.tile(tf.expand_dims(attn_mask, 1),
                                (1, self.n_heads, 1, 1))

        concat, attn_scores = self.scaled_dot_product_attention(query, key, value, attn_mask)

        # B*n_head*L*d -> B*L*d_model
        concat = tf.transpose(concat, (0, 2, 1, 3))
        concat = tf.reshape(concat, (batch_size, -1, self.d_model))
        concat = tf.tensordot(concat, self.W_o, [2, 0])
        output = tf.nn.dropout(concat, rate=self.dropout)
        return output, attn_scores

    def cache_kv(self, x):

        key = tf.tensordot(x, self.W_k, [2, 0])
        value = tf.tensordot(x, self.W_v, [2, 0])
        return [key, value]

    def forward_with_cache(self, query, key, value, attn_mask=None):

        # key and value has been calculated in cache.
        batch_size = tf.shape(query)[0]
        query = tf.tensordot(query, self.W_q, [2, 0])

        # B*L*d_model -> B*L*n_heads*d
        query = tf.reshape(query, shape=(batch_size, -1, self.n_heads, self.d))
        key = tf.reshape(key, shape=(batch_size, -1, self.n_heads, self.d))
        value = tf.reshape(value, shape=(batch_size, -1, self.n_heads, self.d))

        # B*L*n_head*d -> B*n_heads*L*d
        query = tf.transpose(query, (0, 2, 1, 3))
        key = tf.transpose(key, (0, 2, 1, 3))
        value = tf.transpose(value, (0, 2, 1, 3))

        if attn_mask is not None:
            attn_mask = tf.tile(tf.expand_dims(attn_mask, 1),
                                (1, self.n_heads, 1, 1))

        concat, attn_scores = self.scaled_dot_product_attention(query, key, value, attn_mask)

        # B*n_head*L*d -> B*L*d_model
        concat = tf.transpose(concat, (0, 2, 1, 3))
        concat = tf.reshape(concat, (batch_size, -1, self.d_model))
        concat = tf.tensordot(concat, self.W_o, [2, 0])
        output = tf.nn.dropout(concat, rate=self.dropout)
        return output, attn_scores

## Encoder
class EncoderLayer():

    def __init__(self, n_heads, d_model, d_ff, dropout=0, name="encoder_layer"):

        with tf.variable_scope(name):
            self.attention = MultiHeadAttention(n_heads, d_model, dropout, name="attention")
            self.norm1 = LayerNorm(d_model, name="norm_1")
            self.norm2 = LayerNorm(d_model, name="norm_2")
            self.ffn = FeedForward(d_model, d_ff, dropout, name="fnn")

    def __call__(self, enc_output, attn_mask):

        residual = enc_output
        # 1. self-attention
        enc_output, attn_scores = self.attention(enc_output, enc_output, enc_output, attn_mask)

        # 2. add & normal
        enc_output = self.norm1(residual + enc_output)

        # 3. feed forward
        residual = enc_output
        enc_output = self.ffn(enc_output)

        # 4. add & normal
        enc_output = self.norm2(residual + enc_output)

        return enc_output, attn_scores

class Encoder():

    def __init__(self, vocab_size, max_len, n_heads, d_model, d_ff,
                 n_layers, dropout=0, name="encoder"):

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        with tf.variable_scope(name):
            self.embeddings = Embedding(vocab_size, d_model, name="enc_embedding")
            self.pos_embedding = PositionEmbedding(max_len, d_model, name="enc_pos_embedding")
            self.layers = [EncoderLayer(n_heads, d_model, d_ff, dropout, name="layers.%d" % n)
                           for n in range(n_layers)]

    def __call__(self, x, attn_mask):

        enc_self_attn_list = []
        enc_output = self.embeddings(x) + self.pos_embedding(x)
        enc_output = tf.nn.dropout(enc_output, rate=self.dropout)
        for enc_layer in self.layers:
            enc_output, attn_scores = enc_layer(enc_output, attn_mask)
            enc_self_attn_list.append(attn_scores)

        return enc_output, enc_self_attn_list

## Decoder
class DecoderLayer():

    def __init__(self, n_heads, d_model, d_ff, dropout=0, name="decoder_layers"):

        with tf.variable_scope(name):
            self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, name="self_attention")
            self.enc_dec_attention = MultiHeadAttention(n_heads, d_model, dropout, name="enc_dec_attention")
            self.norm1 = LayerNorm(d_model, name="norm1")
            self.norm2 = LayerNorm(d_model, name="norm2")
            self.norm3 = LayerNorm(d_model, name="norm3")
            self.fnn = FeedForward(d_model, d_ff, dropout, name="fnn")

    def __call__(self, enc_output, dec_output, self_attn_mask, enc_dec_attn_mask):

        # 1. self-attention
        residual = dec_output
        dec_output, self_attn_scores = self.self_attention(dec_output,
                                                           dec_output,
                                                           dec_output,
                                                           self_attn_mask)

        # 2. add & norm
        dec_output = self.norm1(residual + dec_output)

        # 3. encoder-deocder attention
        residual = dec_output
        dec_output, enc_dec_attn_scores = self.enc_dec_attention(dec_output,
                                                                 enc_output,
                                                                 enc_output,
                                                                 enc_dec_attn_mask)

        # 4. add & norm
        dec_output = self.norm2(residual + dec_output)

        # 5. fnn - add & norm
        residual = dec_output
        dec_output = self.fnn(dec_output)

        dec_output = self.norm3(residual + dec_output)

        return dec_output, self_attn_scores, enc_dec_attn_scores

    def cache_enc_kv(self, enc_output):

        key, value = self.enc_dec_attention.cache_kv(enc_output)
        return [key, value]

    def cache_dec_kv(self, dec_output):

        key, value = self.self_attention.cache_kv(dec_output)
        return [key, value]

    def forward_with_cache(self, enc_keys, enc_values, dec_enc_attn_mask,
                           dec_keys, dec_values, dec_output):

        residual = dec_output
        dec_output, self_attn_scores = self.self_attention.forward_with_cache(dec_output,
                                                                              dec_keys,
                                                                              dec_values)
        dec_output = self.norm1(residual + dec_output)

        residual = dec_output
        dec_output, enc_dec_attn_scores = self.enc_dec_attention.forward_with_cache(dec_output,
                                                                                    enc_keys,
                                                                                    enc_values,
                                                                                    dec_enc_attn_mask)

        dec_output = self.norm2(residual + dec_output)

        residual = dec_output
        dec_output = self.fnn(dec_output)
        dec_output = self.norm3(residual + dec_output)

        return dec_output, self_attn_scores, enc_dec_attn_scores

class Decoder():

    def __init__(self, vocab_size, max_len, n_heads, d_model,
                 d_ff, n_layers, dropout=0, share_embeddings=None, share_out_proj=True,
                 name = "decoder"):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout

        with tf.variable_scope(name):
            if share_embeddings is not None:
                self.embedding = share_embeddings
            else:
                self.embedding = Embedding(vocab_size, d_model, name="trg_embedding")

            self.pos_embedding = PositionEmbedding(max_len, d_model)
            self.layers = [DecoderLayer(n_heads, d_model, d_ff, dropout,
                                        name="layers.%d" % n) for n in range(n_layers)]

            if share_out_proj:
                self.W = tf.transpose(self.embedding, [1, 0])
            else:
                init_parameters = tf.uniform_unit_scaling_initializer(-1 / np.sqrt(d_model),
                                                                      1 / np.sqrt(d_model))
                self.W = tf.get_variable(name="W",
                                         shape=[d_model, vocab_size],
                                         initializer=init_parameters,
                                         dtype=tf.float32)

    def __call__(self, y, enc_output, self_attn_mask, enc_dec_attn_mask):

        # 1. embedding
        dec_output = self.embedding(y) + self.pos_embedding(y)
        dec_output = tf.nn.dropout(dec_output, rate=self.dropout)

        for i, layer in enumerate(self.layers):
            dec_output, self_attn_scores, enc_dec_attn_scores = layer(enc_output,
                                                                      dec_output,
                                                                      self_attn_mask,
                                                                      enc_dec_attn_mask)
        logits = tf.tensordot(dec_output, self.W, (2, 0))
        logits = tf.reshape(logits, shape=(-1, self.vocab_size))

        return logits


    def cache_dec_kv(self, dec_output):
        kv_list = []
        for dec_layer in self.layers:
            kv_list.append(dec_layer.cache_dec_kv(dec_output))
        return kv_list

    def cache_enc_kv(self, enc_output):
        kv_list = []
        for dec_layer in self.layers:
            kv_list.append(dec_layer.cache_enc_kv(enc_output))
        return kv_list

    def _step(self, y, steps, enc_kv_list, dec_enc_attn_mask, dec_kv_list):

        dec_output = self.embedding(y) + self.pos_embedding(y, steps)

        dec_self_attn_list = []
        dec_enc_attn_list = []
        for i, dec_layer in enumerate(self.layers):
            current_kv = dec_layer.cache_dec_kv(dec_output)
            dec_kv_list[i][0] = tf.concat([dec_kv_list[i][0], current_kv[i][0]], axis=1)
            dec_kv_list[i][1] = tf.concat([dec_kv_list[i][1], current_kv[i][1]], axis=1)
            dec_output, self_attn_scores, dec_enc_attn_scores = \
                dec_layer.forward_with_cache(enc_kv_list[i][0],
                                             enc_kv_list[i][1],
                                             dec_enc_attn_mask,
                                             dec_kv_list[i][0],
                                             dec_kv_list[i][1],
                                             dec_output)
            dec_self_attn_list.append(self_attn_scores)
            dec_enc_attn_list.append(dec_enc_attn_scores)

        logits = tf.tensordot(dec_output, self.W, (2, 0))
        logits = tf.reshape(logits, shape=(-1, self.vocab_size))

        return dec_kv_list, logits, dec_self_attn_list, dec_enc_attn_list

class Transformer():

    def __init__(self, src_vocab_size, src_max_len, trg_vocab_size, trg_max_len,
                 n_heads, d_model, d_ff, n_enc_layers, n_dec_layers,
                 symbols, dropout=0, share_embeddings=True, share_out_proj=True):

        self.PAD = symbols["PAD"]
        self.BOS = symbols["BOS"]
        self.EOS = symbols["EOS"]

        self.src_vocab_size = src_vocab_size
        self.src_max_len = src_max_len
        self.trg_vocab_size = trg_vocab_size
        self.trg_max_len = trg_max_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers

        # encoder
        self.encoder = Encoder(src_vocab_size, src_max_len, n_heads, d_model,
                               d_ff, n_enc_layers, dropout)

        common_embeddings = None
        if share_embeddings:
            common_embeddings = self.encoder.embeddings

        # decoder
        self.decoder = Decoder(trg_vocab_size, trg_max_len, n_heads, d_model,
                               d_ff, n_dec_layers, dropout, common_embeddings, share_out_proj)


    # creat attention mask
    def get_attn_mask(self, query, key):

        query_len = tf.shape(query)[1]
        mask = tf.not_equal(key, self.PAD)
        mask = tf.tile(tf.expand_dims(mask, 1), (1, query_len, 1))
        return mask

    def get_subsequent_mask(self, query):
        # 创建decoder解码self-attention时的mask， 前面的token看不见后面的token
        query_len = tf.shape(query)[1]
        mask = tf.ones(shape=[query_len, query_len], dtype=tf.bool)
        mask = tf.matrix_band_part(mask, -1, 0) # lower triangular part
        return mask

    def __call__(self, x, y):

        # construct attention mask
        enc_self_attn_mask = self.get_attn_mask(x, x)
        dec_self_attn_mask = self.get_attn_mask(y, y)
        dec_self_attn_mask = tf.logical_and(dec_self_attn_mask, self.get_subsequent_mask(y))

        dec_enc_attn_mask = self.get_attn_mask(y, x)

        # cast to float
        enc_self_attn_mask = tf.cast(enc_self_attn_mask, tf.float32)
        dec_self_attn_mask = tf.cast(dec_self_attn_mask, tf.float32)
        dec_enc_attn_mask = tf.cast(dec_enc_attn_mask, tf.float32)

        # encoder & decoder
        enc_output = self.encoder(x, enc_self_attn_mask)
        logits = self.decoder(y, enc_output, dec_self_attn_mask, dec_enc_attn_mask)

        return logits

    def expand_beam_step_with_constraints(self, y, steps, enc_kv_list, dec_kv_list,
                                          dec_enc_attn_mask, last_log_probs, mask_finished,
                                          finished, batch_size, beam_size, hypothesis,
                                          max_decode_steps, repeat_penalty, history_penalty):

        dec_kv_list, logits, _, _ = self.decoder._step(y, steps, enc_kv_list,
                                                       dec_enc_attn_mask, dec_kv_list)

        # mask finished
        finished_mask = tf.cast(finished, tf.float32)
        _log_probs = tf.nn.log_softmax(logits, 1) * (1 - finished_mask)
        _log_probs = _log_probs + mask_finished * finished_mask
        log_probs = last_log_probs[:, -1:] + _log_probs

        log_probs, indice = tf.nn.top_k(log_probs, beam_size)

        log_probs = tf.reshape(log_probs, [-1, 1])
        y = tf.reshape(indice % self.trg_vocab_size, [-1, 1])
        beam_id = tf.tile(tf.expand_dims(tf.range(batch_size), 1),
                          [1, beam_size])
        beam_id = tf.reshape(beam_id, [-1])
        for i in range(self.n_enc_layers):
            enc_kv_list[i][0] = tf.gather(enc_kv_list[i][0], beam_id)
            enc_kv_list[i][1] = tf.gather(enc_kv_list[i][1], beam_id)
            dec_kv_list[i][0] = tf.gather(dec_kv_list[i][0], beam_id)
            dec_kv_list[i][1] = tf.gather(dec_kv_list[i][1], beam_id)
        dec_enc_attn_mask = tf.gather(dec_enc_attn_mask, beam_id)

        log_probs = log_probs - tf.gather(finished_mask, beam_id) * 10000
        log_probs = tf.concat([tf.gather(last_log_probs, beam_id), log_probs], 1)
        finished = tf.logical_or(tf.gather(finished, beam_id), tf.equal(y, self.EOS))

        hypothesis = tf.concat([tf.gather(hypothesis, beam_id), y], axis=1)

        return y, steps+1, enc_kv_list, dec_kv_list, dec_enc_attn_mask, \
               log_probs, finished, hypothesis
