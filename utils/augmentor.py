from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
import random

# 图增强方法
class GraphAugmentor(object):

    @staticmethod
    def node_dropout(sp_adj: csr_matrix, drop_rate):
        # user和item的交互邻接矩阵
        row_num, column_num = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(row_num), int(row_num * drop_rate))
        drop_item_idx = random.sample(range(column_num), int(column_num * drop_rate))
        indicator_user = np.ones(row_num, dtype=np.float32)
        indicator_item = np.ones(column_num, dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(row_num, column_num))
        # diag * matrix * diag
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    @staticmethod
    def edge_dropout(sp_adj: csr_matrix, drop_rate):
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj
