import torch
from torch import nn

class SGL(nn.Module):

    def __init__(self, num_users, num_items, hidden_size, n_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # 构建user和item的embedding信息
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)


    # 根据传入的图的邻接矩阵结构, 进行图卷积操作
    def forward(self, adj_matrix):
        # e_0
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # e_0, e_1, ... e_n
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            if isinstance(adj_matrix, list):
                # 多层的邻接矩阵
                # e_k
                ego_embeddings = torch.sparse.mm(adj_matrix[i], ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            # stack
            all_embeddings += [ego_embeddings]
        # sum
        all_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        # split
        final_user_embedding, final_item_embedding = torch.split(all_embeddings,
                                                                 [self.num_users, self.num_items], dim=0)
        return final_user_embedding, final_item_embedding

    def param_weight(self, indices):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        return ego_embeddings[indices]