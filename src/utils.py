import numpy as np
import torch

def orth(A):
    A_orth, _ = np.linalg.qr(A)
    return A_orth

def norm(A, ord=None):
    if ord is None:
        return np.linalg.norm(A)
    else:
        return np.linalg.norm(A, ord=ord)