import torch
import numpy as np

from torch import nn


class SASRec(nn.Module):

    def __init__(self, num_items, max_len, hidden_size, num_blocks, num_head, dropout_rate, device):
        super().__init__()
        # params
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.device = device
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_head = num_head
        # layer
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.hidden_size,
                                                                      self.num_head,
                                                                      self.hidden_size,
                                                                      self.dropout_rate),
                                                         num_layers=self.num_blocks)

    def forward(self, seq, pos):
        # seq
        seq = torch.LongTensor(seq).to(self.device)
        seq_input = self.item_embedding(seq)
        # 正则化, 为了使sequence和position在同一数量级
        seq_input *= self.hidden_size ** 0.5
        # pos
        pos = torch.LongTensor(pos).to(self.device)
        # 对于seq中存在pad填充的位置, 同样pos也应该放入0进行填充
        pos *= (seq != 0)
        pos_input = self.pos_embedding(pos)
        # input (batch_size, num_steps, hidden_size)
        input = self.embedding_dropout(seq_input + pos_input)
        # 注意力掩码
        # 生成下对角为false的矩阵
        # 矩阵为true代表不参与注意力权重的计算, false参与计算
        # 矩阵的行代表target, 列代表source, 只有target>=source时, 才考虑注意力权重
        att_mask = ~torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool))
        # padding mask
        pad_mask = seq == 0
        # (num_steps, batch_size, hidden_size)
        input = torch.permute(input, (1, 0, 2))
        # transformer encoder
        output = self.transformer_encoder(input,
                                        mask=att_mask.to(self.device),
                                        src_key_padding_mask=pad_mask.to(self.device))
        # (batch_size, num_steps, hidden_size)
        output = torch.permute(output, (1, 0, 2))
        return output


    def get_item_embeddings(self, idx):
        return self.item_embedding(torch.LongTensor(idx).to(self.device))