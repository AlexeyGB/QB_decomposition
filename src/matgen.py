import numpy as np
import torch


def get_with_rank(size, rank=None, use_torch=False, device='cpu'):
    if rank is None:
        rank = min(size)
    if use_torch:
        A1 = torch.randn(size[0], rank).to(device)
        A2 = torch.randn(rank, size[1]).to(device)
        A = torch.matmul(A1, A2).to(device)
    else:
        A1 = np.random.randn(size[0], rank)
        A2 = np.random.randn(rank, size[1])
        A = np.matmul(A1, A2)
    return A