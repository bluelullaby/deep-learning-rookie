import torch
import numpy as np

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_cpu():
    return torch.device('cpu')


# crc的稀疏矩阵转tensor
def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    coords = np.array([coo.row, coo.col])
    i = torch.LongTensor(coords)
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)
