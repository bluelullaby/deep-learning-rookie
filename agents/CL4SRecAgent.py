import os

import torch
import shutil
import numpy as np
import torch.nn.functional as F

from torch import nn
from base.base_trainer import BaseTrainer
from data_loader.recommender import Sequence, load_data_file
from data_loader.amazon import next_batch_sequence
from data_loader.amazon import next_batch_sequence_for_test
from model.SASRec import SASRec
from utils.util import find_k_largest
from utils.evaluation import Metric
from utils.augmentor import SequenceAugmentor



class CL4SRecAgent(BaseTrainer):

    def __init__(self,
                 data_dir="./datasets/amazon-beauty",
                 batch_size=256,
                 max_len=50,
                 hidden_size=64,
                 num_blocks=2,
                 num_head=1,
                 dropout_rate=0.2,
                 lr=0.001,
                 reg_coefficient=0.0001,
                 cl_coefficient=0.1,
                 temperature=1,
                 max_top=20,
                 aug_type='crop',
                 aug_rate=0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # params
        self.batch_size = batch_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_head = num_head
        self.dropout_rate = dropout_rate
        self.reg_coefficient = reg_coefficient
        self.cl_coefficient = cl_coefficient
        self.lr = lr
        self.max_top = max_top
        self.aug_type = aug_type
        self.aug_rate = aug_rate
        self.temperature = temperature
        # dataset
        self.data = Sequence(load_data_file(os.path.join(data_dir, "train.txt"), data_type="sequence"),
                             load_data_file(os.path.join(data_dir, "test.txt"), data_type="sequence"))
        # model
        # mask augment时候, 需要增加1个长度
        self.model = SASRec(self.data.item_num + 1,
                            self.max_len,
                            self.hidden_size,
                            self.num_blocks,
                            self.num_head,
                            self.dropout_rate,
                            self.device).to(self.device)

        # loss
        self.loss = nn.BCEWithLogitsLoss()

        # trainer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mnt_best = None
        self.early_stop = None if "early_stop" not in kwargs else kwargs["early_stop"]


    # 序列增强
    def sequence_reconstruction(self, seq, pos, seq_len):
        if self.aug_type == 'crop':
            # 裁剪
            seq, pos, seq_len = SequenceAugmentor.item_crop(seq, seq_len, self.aug_rate)
        elif self.aug_type == 'reorder':
            # 重排序
            seq = SequenceAugmentor.item_reorder(seq, seq_len, self.aug_rate)
        elif self.aug_type == 'mask':
            # 掩码
            seq = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate, self.data.item_num + 1)
        return seq, pos, seq_len


    # InfoNCE
    def infonce_loss(self, view1, view2, temperature: float, b_cos: bool = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()


    def train_one_epoch(self):
        for batch_idx, (seq, pos, y, neg_idx, seq_len) in enumerate(next_batch_sequence(self.data,
                                                                                        self.batch_size,
                                                                                        self.max_len)):
            # 经过SASRec模型
            ffn_embedding = self.model(seq, pos)
            # 序列增强
            aug_seq1, aug_pos1, aug_len1 = self.sequence_reconstruction(seq, pos, seq_len)
            aug_seq2, aug_pos2, aug_len2 = self.sequence_reconstruction(seq, pos, seq_len)
            # 增强序列同样经过Transformer模型
            aug_ffn_embedding1 = self.model(aug_seq1, aug_pos1)
            aug_ffn_embedding2 = self.model(aug_seq2, aug_pos2)
            # 增强序列中获得每一个batch的最后一个时间步的embedding表示
            aug_last_embeddings1 = [aug_ffn_embedding1[index, len - 1, :].reshape((-1, self.hidden_size)) for index, len in enumerate(aug_len1)]
            aug_last_embeddings2 = [aug_ffn_embedding2[index, len - 1, :].reshape((-1, self.hidden_size)) for index, len in enumerate(aug_len2)]
            # 计算InfoNCE
            batch_loss = self.cl_coefficient * self.infonce_loss(torch.cat(aug_last_embeddings1, dim=0),
                                                                 torch.cat(aug_last_embeddings2, dim=0),
                                                                 self.temperature)
            # 计算主流程的损失
            ground_truth = self.model.get_item_embeddings(y)
            negative_value = self.model.get_item_embeddings(neg_idx)
            pos_logits, neg_logits = (ground_truth * ffn_embedding).sum(dim=-1), (negative_value * ffn_embedding).sum(dim=-1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape,
                                                                                                   device=self.device)
            mask = np.where(pos != 0)
            batch_loss += self.loss(pos_logits[mask], pos_labels[mask])
            batch_loss += self.loss(neg_logits[mask], neg_labels[mask])
            for param in self.model.item_embedding.parameters():
                batch_loss += torch.norm(param, p=2) * self.reg_coefficient
                # 梯度清零
                self.optimizer.zero_grad()
                # 反向梯度
                batch_loss.backward()
                self.optimizer.step()
                # logging
                if batch_idx % self.log_interval == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.current_epoch, batch_idx * len(seq), len(self.data.training_data),
                                            100. * (batch_idx * len(seq) / len(self.data.training_data)), batch_loss))


    def validate(self):
        rec_list = {}
        # 每一条sequence的最后50个数据
        for batch_idx, (seq, pos, seq_len) in enumerate(next_batch_sequence_for_test(self.data, self.batch_size, self.max_len)):
            seq_start = batch_idx * self.batch_size
            seq_end = (batch_idx + 1) * self.batch_size
            # user
            seq_names = [seq_full[0] for seq_full in self.data.original_seq[seq_start:seq_end]]
            # 得到每一个batch对应的全部的item的权重
            candidates = self.predict(seq, pos, seq_len)
            for name, candidate in zip(seq_names, candidates):
                ids, scores = find_k_largest(self.max_top, candidate)
                item_names = [self.data.id2item[iid] for iid in ids if iid != 0 and iid <= self.data.item_num]
                rec_list[name] = list(zip(item_names, scores))
        return self.ranking_evaluation(self.data.test_set, rec_list, [self.max_top])


    def predict(self, seq, pos, seq_len):
        with torch.no_grad():
            # 经过transformer后的ffn的embedding信息
            ffn_embedding = self.model(seq, pos)
            # 每一个batch中对应len-1位置的那个时间步的embedding表示
            # [(1, hidden_size)]
            last_item_embeddings = [ffn_embedding[index, len - 1, :].reshape((-1, self.hidden_size)) for index, len in enumerate(seq_len)]
            # 按dim=0拼接
            # (batch_size, hidden_size)
            last_item_embeddings = torch.concat(last_item_embeddings, dim=0)
            # 按照train过程中的MF原理
            # 用ffn得到的embedding信息与具体的item的embedding点积求和得到对应item的权重
            # 这里拿出所有的item信息, 得到每一个batch对应的所有item的加权和
            # (batch_size, item_nums)
            score = torch.matmul(last_item_embeddings, self.model.item_embedding.weight.T)
            return score.cpu().numpy()


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

            improved = self.mnt_best is None or (
                sum(1 if self.mnt_best[k] > performance[k] else -1 for k in performance)) < 0

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