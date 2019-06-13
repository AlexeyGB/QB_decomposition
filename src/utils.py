import numpy as np
import torch

def orth(A):
    A_orth, _ = np.linalg.qr(A)
    return A_orth