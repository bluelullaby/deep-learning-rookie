import logging
import os
import shutil

import torch
from torch import nn
from utils.util import *
from base.base_trainer import BaseTrainer
from data_loader.recommender import load_text_file
from data_loader.recommender import Interaction
from data_loader.yelp import next_batch_pairwise
from model.LightGCN import LightGCN
from utils.util import convert_sparse_mat_to_tensor



class LightGCNAgent(BaseTrainer):

    def __init__(self,
                 data_dir="./datasets/yelp2018",
                 batch_size=64,
                 lr=0.1,
                 embedding_size=128,
                 n_layers=2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = Interaction(load_text_file(os.path.join(data_dir, "train.txt")),
                                load_text_file(os.path.join(data_dir, "test.txt")))
        self.norm_adj = convert_sparse_mat_to_tensor(self.data.norm_adj)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.model = LightGCN(self.data.user_num, self.data.item_num, embedding_size, n_layers, self.norm_adj,
                              self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # 根据user的embedding信息和正负item的embedding信息, 计算损失函数
    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        # user_emb - (batch_size * hidden_size)
        # pos_item_emb - (batch_size * hidden_size)
        # 每一个做点积操作, 得到一个user对特定一个item的embedding表示
        # 最终求和得到加权结果
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def l2_loss(self, reg, *params):
        loss = 0.0
        for param in params:
            loss += torch.pow(param, 2).sum()
        return reg * loss

    def train_one_epoch(self):
        # user, position, negative
        for user_idx, pos_idx, neg_idx in next_batch_pairwise(self.data, self.batch_size):
            # 得到更新后的user和item的embedding信息
            user_embeddings, item_embeddings = self.model(self.data)
            # 分别得到batch对应的embedding信息
            user_emb, pos_item_emb, neg_item_emb = user_embeddings[user_idx], item_embeddings[pos_idx], item_embeddings[
                neg_idx]
            # 计算损失
            bpr_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb).mean()
            reg_loss = self.l2_loss(0.5, user_emb, pos_item_emb, neg_item_emb)
            loss = bpr_loss + reg_loss
            # 计算梯度
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
