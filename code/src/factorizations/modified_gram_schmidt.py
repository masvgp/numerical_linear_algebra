import numpy as np


def mgs(A):
    """An implementation of modified Gram-Schmidt for QR factorization. The implementation uses orthogonal projections
    to compute the orthonormal vectors q that form the columns of the matrix Q.

    This is the modified Gram-Schmidt Algorithm as presented in Algorithm 8.1 in the book 'Numerical Linear Algebra' by Trefethen and Bau

    Args:
        A (arr): An m x n matrix A
    Output:
        Q (arr): an m x n matrix with orthonormal columns
        R (arr): an n x n upper-diagonal matrix
    """
    m = A.shape[0] # Get row-dim of A
    n = A.shape[1] # Get col-dim of A
    Q = np.zeros((m, n)) # Initialize matrix Q
    R = np.zeros((n, n)) # Initialize matrix R
    
    # Copy the matrix A into V. This is a loop in Algorithm 8.1,
    # but the loop is unecessary
    V = A.copy().astype(np.float64)

    for i in range(n):
        # Raise error if provided matrix is singular
        if np.linalg.norm(V[:, i]) == 0:
                raise ValueError("The provided matrix is singular. Modified Gram-Schmidt only works for non-singular matrices")

        R[i, i] = np.linalg.norm(V[:, i]) # Compute each r_ii
        Q[:, i] = V[:, i] / R[i, i] # normalize each orthogonal v
        
        for j in range(i+1, n):
            R[i, j] = Q[:, i]@V[:, j] # Compute r_ij
            V[:, j] = V[:, j] - R[i, j]*Q[:, i] # Get the orthogonal projection as soon as the latest
                                                # q_i is known
            
            
    
    return Q, R