import numpy as np
import torch

def randQB(A, k, s, use_torch=False, device='cpu'):
    '''
    A: matrix
    k: rank
    s: small int
    '''
    m, n = A.shape
    
    if torch:
        raise NotImplementedError
    else:
        omega = np.random.randn(n, k + s)
        #Q = 