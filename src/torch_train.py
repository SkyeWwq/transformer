# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 5:15 下午
# @Author  : Dawein
# @File    : torch_train.py
# @Software : PyCharm

"""
模型训练脚本
"""

import os
import time
import logging
import torch
import torch.nn.functional as F
from torch import optim
from torch_transformer import Transformer
from data_generator import Vocab, DateGenerator
from utils import Utils

class Train:
    def __init__(self, param_config):
        self.pConfig = param_config
        self.vocab = Vocab(self.pConfig.vocab_file)

    # calc loss
    def calc_loss(self, logits, target, eps, pad):

        n_class = logits.size(1)
        if eps > 0:
            # label smoothing
            one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.sum(log_probs * one_hot, dim=1)
            mask = target.view(-1).ne(pad)
            loss = loss * mask.type(torch.float)
            loss = torch.sum(loss) / torch.sum(mask.type(torch.float))
        else:
            # cross entropy
            loss = F.cross_entropy(input=logits, target=target.view(-1), ignore_index=pad)

        return loss

    def train(self):

        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_str)
        logger = logging.getLogger(__name__)

        # get parameters
        use_cuda = self.pConfig.use_cuda
        enc_layer_num = self.pConfig.enc_layer_num
        dec_layer_num = self.pConfig.dec_layer_num
        n_heads = self.pConfig.n_heads
        d_model = self.pConfig.d_model
        d_ff = self.pConfig.d_ff
        dropout = self.pConfig.dropout
        src_max_len = self.pConfig.src_max_len
        trg_max_len = self.pConfig.trg_max_len
        src_vocab_size = self.vocab.size
        trg_vocab_size = self.vocab.size
        share_embeddings = self.pConfig.share_embeddings
        share_embeddings_proj = self.pConfig.share_embeddings_prob

        symbols = {"PAD": self.vocab.token2idx[self.pConfig.PAD],
                        "BOS": self.vocab.token2idx[self.pConfig.BOS],
                        "EOS": self.vocab.token2idx[self.pConfig.EOS],
                        "UNK": self.vocab.token2idx[self.pConfig.UNK]}

        # create model
        model = Transformer(src_vocab_size, src_max_len,
                            trg_vocab_size, trg_max_len,
                            n_heads, d_model, d_ff, enc_layer_num, dec_layer_num,
                            symbols, dropout, share_embeddings, share_embeddings_proj)
        if use_cuda:
            model = model.cuda()

        total_aprams = sum(p.numel() for p in model.parameters())
        logger.info("Total parameters of model: {}".format(total_aprams))

        optimizer = optim.Adam(model.parameters(), lr=self.pConfig.lr, amsgrad=True)

        # pre-shuffle
        train_file = self.pConfig.train_file
        utils = Utils()
        utils.shuffle_data(file_in=train_file)
        logger.info("Shuffle finished.")

        # training
        batcher = DateGenerator(self.vocab.token2idx, symbols, src_max_len, trg_max_len)
        epoches = self.pConfig.epochs
        save_steps = self.pConfig.save_steps
        batch_size = self.pConfig.batch_size
        model_dir = self.pConfig.model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        logger.info("Start to training...")
        all_steps = 0
        start_time = time.time()
        for e in range(1, epoches + 1):
            steps_every_epoch = 0
            for batch in batcher.batcher(train_file, batch_size, batch_num=1):
                if use_cuda:
                    x = torch.tensor(batch[0], dtype=torch.long).cuda()
                    y = torch.tensor(batch[1], dtype=torch.long).cuda()
                    target_y = torch.tensor(batch[2], dtype=torch.long).cuda()
                else:
                    x = torch.tensor(batch[0], dtype=torch.long)
                    y = torch.tensor(batch[1], dtype=torch.long)
                    target_y = torch.tensor(batch[2], dtype=torch.long)

                optimizer.zero_grad()

                logits = model(x, y)
                loss = self.calc_loss(logits, target_y, eps=0, pad=symbols["PAD"])
                logger.info("epoch {} step {}, loss: {:.5f}".format(e, steps_every_epoch, loss))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.pConfig.grad_clip)

                optimizer.step()

                # save model
                if all_steps > 0 and all_steps % save_steps == 0:
                    logger.info("Save model to %s" % (model_dir))
                    model_name = "{}_{}.pkl".format("model", all_steps)
                    optimizer_name = "{}_{}.optimizer".format("model", all_steps)

                    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
                    torch.save(optimizer.state_dict(), os.path.join(model_dir, optimizer_name))

                # add
                steps_every_epoch += 1
                all_steps += 1

        # calc time
        end_time = time.time()
        m, s = divmod(end_time - start_time, 60)
        h, m = divmod(m, 60)
        logger.info("Training finished, total cost: %d:%02d:%02d" % (h, m, s))