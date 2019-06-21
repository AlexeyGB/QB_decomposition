from copy import deepcopy

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


def randQB_b(A, rel_err, b, p=0, use_torch=False, device='cpu'):
    '''
    A: matrix
    rel_err: relative error of decomposition
    b: block size
    p: power parameter, (default 0)
    '''
    m, n = A.shape
    
    if use_torch:
        raise NotImplementedError
    else:
        Q = None
        B = None
        
        i = 0
        while norm(A) >= rel_err:
            Omega_i = np.random.randn(n, b)
            Q_i = orth(np.matmul(A, Omega_i))
            
            # power iteration
            for j in range(1, p+1):
                Q_i = orth(np.matmul(A.T, Q_i))
                Q_i = orth(np.matmul(A, Q_i))
            if i != 0:    
                Q_i = orth(Q_i - np.matmul(Q, np.matmul(Q.T, Q_i)))
                
            B_i = np.matmul(Q_i.T, A)
            A = A - np.matmul(Q_i, B_i)
            
            if i == 0:
                Q = Q_i
                B = B_i
            else:
                Q = np.hstack((Q, Q_i))
                B = np.vstack((B, B_i))
                
            i += 1
    return Q, B


def randQB_EI(A, rel_err, b, p=0, use_torch=False, device='cpu'):
    '''
    A: matrix
    rel_err: relative error of decomposition
    b: block size
    p: power parameter, (default 0)
    '''
    m, n = A.shape
    threshold = rel_err ** 2
    
    if use_torch:
        raise NotImplementedError
    else:
        Q = None
        B = None
        
        E = norm(A) ** 2
        i = 0
        while E >= threshold:
            Omega_i = np.random.randn(n, b)
            if i != 0:
                Q_i = orth(np.matmul(A, Omega_i) - np.matmul(Q, np.matmul(B, Omega_i)))
                
                # power iteration
                for j in range(1, p+1):
                    Q_i = orth(np.matmul(A.T, Q_i) - np.matmul(B.T, np.matmul(Q.T, Q_i)))
                    Q_i = orth(np.matmul(A, Q_i) - np.matmul(Q, np.matmul(B, Q_i)))
                    
                Q_i = orth(Q_i - np.matmul(Q, np.matmul(Q.T, Q_i)))
            if i == 0:
                Q_i = orth(np.matmul(A, Omega_i))
                
                # power iteration
                for j in range(1, p+1):
                    Q_i = orth(np.matmul(A.T, Q_i))
                    Q_i = orth(np.matmul(A, Q_i))
                
            B_i = np.matmul(Q_i.T, A)
            
            if i == 0:
                Q = Q_i
                B = B_i
            else:
                Q = np.hstack((Q, Q_i))
                B = np.vstack((B, B_i))
            
            E = E - norm(B_i) ** 2
            i += 1
    return Q, B
                
    
def greedyQB(A, rel_err):
    '''
    A: matrix
    rel_err: relative error of decomposition
    '''
    
    m, n = A.shape
    Q = None
    B = None
    
    i = 0
    curr_err = rel_err * 2
    while curr_err > rel_err:
        idx = np.random.randint(n, size=1)
        q_i = A[:, idx]
        q_i = q_i / norm(q_i)
        b_i = np.matmul(q_i.T, A)
        
        if i == 0:
            Q = q_i
            B = b_i
        else:
            Q = np.hstack((Q, q_i))
            B = np.vstack((B, b_i))
            
        A = A - np.matmul(q_i, b_i)
        curr_err = norm(A)
        i += 1
    
    return Q, B


def SVD_from_QB(A, QB_method=randQB, *args, **kwargs):
    Q, B = QB_method(A, *args, **kwargs)
    U_, S, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_
    return U, S, Vh
