import logging
import os
import shutil

import torch
from torch import nn
from utils.util import *
from base.base_trainer import BaseTrainer
from data_loader.recommender import load_data_file
from data_loader.recommender import Interaction
from data_loader.yelp import next_batch_pairwise
from model.LightGCN import LightGCN
from utils.util import convert_sparse_mat_to_tensor
from utils.evaluation import Metric
from scipy import sparse


class LightGCNAgent(BaseTrainer):

    def __init__(self,
                 data_dir="./datasets/yelp2018",
                 batch_size=2048,
                 lr=0.001,
                 embedding_size=64,
                 n_layers=3,
                 max_top=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = Interaction(load_data_file(os.path.join(data_dir, "train.txt")),
                                load_data_file(os.path.join(data_dir, "test.txt")))
        self.norm_adj = convert_sparse_mat_to_tensor(self.data.norm_adj).to(self.device)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.max_top = max_top
        self.model = LightGCN(self.data.user_num, self.data.item_num, embedding_size, n_layers, self.norm_adj).to(self.device)
        self.mnt_best = None
        self.early_stop = None if "early_stop" not in kwargs else kwargs["early_stop"]
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

    def l2_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.pow(emb, 2).sum()
        return emb_loss * reg


    def train_one_epoch(self):
        # user, position, negative
        for batch_idx, (user_idx, pos_idx, neg_idx) in enumerate(next_batch_pairwise(self.data, self.batch_size)):
            # 得到更新后的user和item的embedding信息
            user_embeddings, item_embeddings = self.model(self.data)
            self.user_embeddings = user_embeddings
            self.item_embeddings = item_embeddings
            # 分别得到batch对应的embedding信息
            user_emb, pos_item_emb, neg_item_emb = user_embeddings[user_idx], item_embeddings[pos_idx], item_embeddings[neg_idx]
            # 计算损失
            bpr_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            reg_loss = self.l2_loss(0.0001,
                                    self.model.param_weight([*user_idx, *pos_idx, *neg_idx])) / self.batch_size
            loss = bpr_loss + reg_loss
            # 计算梯度
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(user_idx), len(self.data.training_data),
                                        100. * (batch_idx * len(user_idx) / len(self.data.training_data)), loss))

    def run(self):
        not_improved_count = 0

        # resume
        if self.resume:
            self.load_checkpoint()

        # train for epochs
        for epoch in range(self.current_epoch + 1, self.max_epoch + 1):
            self.current_epoch = epoch

            self.model.train()
            self.train_one_epoch()

            self.model.eval()
            measure = self.validate()
            performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

            improved = self.mnt_best is None or (sum(1 if self.mnt_best[k] > performance[k] else -1 for k in performance)) < 0

            if improved:
                self.mnt_best = performance
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1
                is_best = False

            self.logger.info('-' * 80)
            self.logger.info(f'Real-Time Ranking Performance (Top-{self.max_top} Item Recommendation)')
            measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
            self.logger.info(f'*Current Performance*\nEpoch: {self.current_epoch}, {measure_str}')
            bp = ', '.join([f'{k}: {v}' for k, v in self.mnt_best.items()])
            self.logger.info(f'*Best Performance*\nEpoch: {self.current_epoch}, {bp}')
            self.logger.info('-' * 80)

            if self.early_stop and not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

            self.save_checkpoint(is_best)


    def predict(self, user):
        # 对某一个用户进行预测
        uid = self.data.get_user_id(user)
        # 用户的embedding向量 (hidden_size)
        uid_emb = self.user_embeddings[uid]
        # 所有item的embedding (item_size, hidden_size)
        item_emb = self.item_embeddings
        score = torch.matmul(item_emb, uid_emb)
        return score.detach().cpu().numpy()


    def validate(self):
        rec_list = {}
        for i, user in enumerate(self.data.test_set):
            # 当前用户对于所有item的评分
            candidates = self.predict(user)
            rated_list, _ = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_top, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
        return self.ranking_evaluation(self.data.test_set, rec_list, [self.max_top])


    def ranking_evaluation(self, origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in res:
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print('The Lengths of test set and predicted set do not match!')
                exit(-1)
            hits = Metric.hits(origin, predicted)
            hr = Metric.hit_ratio(origin, hits)
            indicators.append('Hit Ratio:' + str(hr) + '\n')
            prec = Metric.precision(hits, n)
            indicators.append('Precision:' + str(prec) + '\n')
            recall = Metric.recall(hits, origin)
            indicators.append('Recall:' + str(recall) + '\n')
            # F1 = Metric.F1(prec, recall)
            # indicators.append('F1:' + str(F1) + '\n')
            # MAP = Measure.MAP(origin, predicted, n)
            # indicators.append('MAP:' + str(MAP) + '\n')
            NDCG = Metric.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            # AUC = Measure.AUC(origin,res,rawRes)
            # measure.append('AUC:' + str(AUC) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure


    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        torch.save(state, os.path.join(self.checkpoint_dir, self.checkpoint_file))
        if is_best:
            shutil.copyfile(os.path.join(self.checkpoint_dir, self.checkpoint_file),
                            os.path.join(self.checkpoint_dir, "model_best.pth.tar"))

    def load_checkpoint(self):
        fname = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        if not os.path.exists(fname):
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
            return
        self.logger.info("Loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)

        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.mnt_best = checkpoint['monitor_best']

    def finalize(self):
        self.save_checkpoint()
