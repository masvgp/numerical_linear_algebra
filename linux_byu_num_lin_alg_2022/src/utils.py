# import sys
# sys.path.insert(0, "/home/masvgp/dev/byu_num_lin_alg_2022/src/")

"""Utility functions
"""
import numpy as np

def e(i, length):
    """Generate a unit coordinate vector, i.e., Given an index i and a length, return a vector with a 1.0 in the ith position and 0.0 else.

    Args:
        i (int): Integer index in which 1.0 will be assigned  
        length (int): Length of the desired output vector

    Returns:
        arr: Unit coordinate vector with 1.0 in the ith position
    """
    e_i = np.zeros(length, dtype=np.float64)
    e_i[i] = 1.0
    return e_i

def vandermonde(x, n):
    """A Vandermonde matrix V of size len(x) x len(x)

    Args:
        x (arr): A vector x of data from which the Vandermonde matrix is to be constructed.
        n (int): Degree of the polynomial represented by V
    """
    # m = len(x) # Number of rows for V
    
    # Construct Vandermonde of data x
    V = np.array([x**k for k in range(n)]).T

    # Return Vandermonde
    return V
