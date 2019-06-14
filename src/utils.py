import numpy as np
import torch

from time import time

def orth(A):
    A_orth, _ = np.linalg.qr(A)
    return A_orth

def norm(A, ord=None):
    if ord is None:
        return np.linalg.norm(A)
    else:
        return np.linalg.norm(A, ord=ord)
    
def time_task(func, metric, num_starts=1, A, *args, **kwargs):
    '''
    Runs task several times, calculates metric 
    func: function to run
    metric: metric to calculate. gets matrix A and value, returned
        by func, as arguments
    A: matrix
    args, kwargs: arguments for func
    '''
    runtimes = np.empty(shape=(num_starts, ), dtype=np.float64)
    metrics =  np.empty(shape=(num_starts, ), dtype=np.float64)
    for i in range(runtimes):
        time_start = time()
        ret_val = func(A, *args, **kwargs)
        runtimes[i] = time() - time_start
        metrics[i] = metric(A, *ret_val)
        
    return runtimes, metrics

def QB_error(A, Q, B):
    A_reconstr = np.matmul(Q, B)
    return norm(A - A_reconstr)

def SVD_error(A, U, S, Vh):
    A_reconstr = U * S @ Vh
    return norm(A - A_reconstr)