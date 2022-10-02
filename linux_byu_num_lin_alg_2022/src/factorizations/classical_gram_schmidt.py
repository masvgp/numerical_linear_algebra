# Classical Gram-Schmidt algorithm
import numpy as np

def cgs(A):
    """Classical gram schmidt computes the QR factorization of a matrix

    Args:
        x (arr): An m x n array of floats
    """
    # Set m and n
    m = A.shape[0]
    n = A.shape[1]
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v_j = A[:, j] # Set v_j = a_j; This initializes v_j when j = 0
        
        # This inner loop will not start executing until j-1 >= 2
        # This inner loop is designed to compute the off-diagonal entries of R
        # Note that we've used j in the range function instead of j-1 to account for Python's indexing
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j] # Compute the off-diagonal entries of Q
            v_j = v_j - R[i, j]*Q[:, i]
        
        R[j, j] = np.linalg.norm(v_j) # Compute the jth diagonal entry of R
        
        Q[:, j] = v_j/R[j, j] # Compute the jth q, after the j-1 subtractions happen in the inner loop
                       # This initializes Q when j = 0
        
    return Q, R