# import sys
# sys.path.insert(0, "/home/masvgp/dev/byu_num_lin_alg_2022/src/")
import numpy as np
from utils import e

def reflector(v):
    """Given a unit vector v, return the householder reflector I - 2 * np.outer(v, v)

    Args:
        v (arr): Vector
    Returns:
        reflector (arr): A len(v) x len(v) array representing a householder reflector
    """
    I = np.eye(len(v))
    outer = np.outer(v, v)


    return (I - 2 * outer)

def formQ(W, reduced=False):
    """Form Q from the lower diagonal vk's formed in the householder algorithm.

    Args:
        W (arr): An m x n matrix containing the vk in the kth column

    Returns:
        arr: Returns an m x m orthonormal matrix Q such that QR = A
    """
    m = W.shape[0]
    n = W.shape[1]
    Q = np.eye(m, m)
    for k in range(n):
            Qk = np.eye(m, m)
            Qk[k:m, k:m] = reflector(W[k:m, k])
            Q = Q @ Qk
    
    if reduced == True:
        # Return reduced Q
        return np.around(Q[:m, :n], decimals=8)
    else:
        # Return full Q
        return np.around(Q, decimals=6)

def house(A, reduced=False):
    """Compute the factor R of a QR factorization of an m x n matrix A with m >= .

    Args:
        A (arr): A numpy array of shape m x n
    Returns:
        R (arr): The upper diagonal matrix R in a QR factorization
    """
    m = A.shape[0] # Get row-dim of A
    n = A.shape[1] # Get col-dim of A
    W = np.zeros((m, n))
    #Q = np.eye(m, m)
    # V = np.zeros((m, n))

    # Cast A as type np.float64
    A = A.astype(np.float64)

    for k in range(n):
        # Get the first column of the (m-k+1, n-k+1) submatrix of A
        x = A[k:m, k]

        # Compute e_1, sign function, and the norm of x
        e_1 = e(0, len(x))
        sign = np.sign(x[0])
        norm_x = np.linalg.norm(x)

        # Compute vk, the vector reflected across the Householder hyperplane
        vk = (sign * norm_x * e_1) + x
        vk = vk / np.linalg.norm(vk)

        # Store vk in W
        W[k:m, k] = vk

        # Apply the reflector to the k:m, k:n submatrix of A to put
        # zeros below the diagonal of the kth column of A
        A[k:m, k:n] = reflector(vk) @ A[k:m, k:n]
    
    if reduced == True:
        return np.around(A[:n, :n], decimals=8), W
    else:
        # Return full R
        return np.around(A, decimals=8), W


# Test code 
# A = np.random.normal(loc=0.0, scale=1.0, size=(10, 6))
# house(A)