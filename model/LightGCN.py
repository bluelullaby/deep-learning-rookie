import torch
from torch import nn
from data_loader.recommender import Interaction


class LightGCN(nn.Module):

    # data - user和item的交互记录
    # emb_size - 隐藏单元大小
    def __init__(self, num_users, num_items, emb_size, n_layers, norm_adj, device):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        # user&item embedding
        self.user_embeddings = nn.Embedding(num_users, self.emb_size)
        self.item_embeddings = nn.Embedding(num_items, self.emb_size)
        # 正则化的邻接矩阵
        self.norm_adj = norm_adj

    def forward(self, data: Interaction):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
        # TODO 原逻辑代码有误
        # 所有层的embedding参数加权求平均
        all_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        # 切分出具体的user和item嵌入信息
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        # 返回嵌入结果
        # (num_users, hidden_size)
        return user_embeddings, item_embeddings
