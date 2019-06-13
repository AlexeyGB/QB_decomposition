import numpy as np
import torch

from .utils import orth, norm

def randQB(A, k, s, use_torch=False, device='cpu'):
    '''
    A: matrix
    k: rank
    s: small int
    '''
    m, n = A.shape
    
    if use_torch:
        raise NotImplementedError
    else:
        Omega = np.random.randn(n, k + s)
        Q = orth(np.matmul(A, Omega))
        B = np.matmul(Q.T, A)
        
    return Q, B