import numpy as np
import torch


def get_with_rank(size, rank, torch=False, device='cpu'):
    if torch:
        A1 = np.random.randn(size[0], rank)
        A2 = np.random.randn(rank, size[1])
        A = np.matmul(A1, A2)
    else:
        A1 = torch.randn(size[0], rank).to(device)
        A2 = torch.randn(rank, size[1]).to(device)
        A = torch.matmul(A1, A2).to(device)
    return A