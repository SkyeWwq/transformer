# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 5:21 下午
# @Author  : Dawein
# @File    : params_config.py
# @Software : PyCharm

"""
参数
"""

import argparse

class ParamsConfig:

    parser = argparse.ArgumentParser()

    # flag token
    parser.add_argument("--PAD", default="<PAD>", type=str)
    parser.add_argument("--BOS", default="<BOS>", type=str)
    parser.add_argument("--EOS", default="<EOS>", type=str)
    parser.add_argument("--UNK", default="<UNK>", type=str)

    # cuda
    parser.add_argument("--use_cuda", default=False, type=bool, help="gpu or cpu")

    # file path
    parser.add_argument("--vocab_file", default="../data/couplet/vocab.txt",
                        type=str, help="vocab file")
    parser.add_argument("--train_file", default="../data/couplet/train.txt",
                        type=str, help="train file")

    # parameters
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--src_max_len", default=30, type=int, help="maximum length of source")
    parser.add_argument("--trg_max_len", default=20, type=int, help="maximum length of target")

    parser.add_argument("--enc_layer_num", default=2, type=int, help="number of encoder layer")
    parser.add_argument("--dec_layer_num", default=2, type=int, help="number of decoder layer")
    parser.add_argument("--n_heads", default=2, type=int, help="number of multi-head")
    parser.add_argument("--d_model", default=16, type=int, help="size of feature")
    parser.add_argument("--d_ff", default=32, type=int, help="size of feed forwar feature")
    parser.add_argument("--dropout", default=0.0, type=float, help="prob of dropout")
    parser.add_argument("--share_embeddings", default=True, type=bool,
                        help="whether share embeddings between encoder and decoder")
    parser.add_argument("--share_embeddings_prob", default=True, type=bool,
                        help="whether share embeddings for final output projection")

    parser.add_argument("--epochs", default=200, type=int, help="epoch of training")
    parser.add_argument("--save_steps", default=199, type=int, help="each steps to save a model")
    parser.add_argument("--lr", default=0.01, type=float, help="learning ratio")
    parser.add_argument("--grad_clip", default=10.0, type=float, help="clip gradient")

    parser.add_argument("--model_dir", default="../model/", type=str, help="dir of saving model")
    parser.add_argument("--model_name", default="model_199.pkl",
                        type=str, help="model name to load")

    parser.add_argument("--beam_size", default=3, type=int, help="beam size")