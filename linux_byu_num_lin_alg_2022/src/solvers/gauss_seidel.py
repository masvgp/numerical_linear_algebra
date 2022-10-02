# Gauss-Seidel method for solving Ax=b.
import numpy as np


def gauss_seidel(A, b, x_start, epsilon=1e-10, max_iterations=500):
    """
    Gauss-Seidel method for solving the system of equations Ax=b.
    A : matrix
        Numpy array containing the numbers in the coefficient matrix.
    b : vector
        Vector containing the numbers on the right-hand side of Ax=b.
    x_start : vector
        A starting guess of the solution vector x.
    epsilon : number
        The tolerance used to determine convergence, Cauchy style.
    max_iterations : number
        Maximum number of iterations to complete.
    returns : vector
        Function returns a solution vector x.
    returns : vector
        Function returns a solution vector x.
    """
    # Get dim(A)
    m = A.shape[0]
    n = A.shape[1]

    if len(b) != len(A):
        raise ValueError("Number of rows of A must match number of rows of b.")
    
    D = np.eye(m, n) @ np.diag(A)
    M = A - D
    D_inv = np.diag(1 / np.diag(D))
    b = np.dot(D_inv, b)
    M = np.dot(D_inv, M)
    x = x_start
    for i in range(max_iterations):
        x_next = x.copy()
        for j in range(A.shape[0]):
            x_next[j] = b[j] - np.dot(M[j, :], x_next)

        if np.linalg.norm(x - x_next) < epsilon:
            return x
        x = x_next
    return x


# Testing using the Jacobi example from Wikipedia.
# This example converges in 14 iterations compared to the 69 iterations taken in the Jacobi method.
# A = np.array([
#     [5, 2, 1, 1],
#     [2, 6, 2, 1],
#     [1, 2, 7, 1],
#     [1, 1, 2, 8]
# ])
# b = np.array([29, 31, 26, 19])
# # you can choose any starting vector
# x = np.zeros(len(b))

# x_result = gauss_seidel(A, b, x_start=x)

# print('The result of gauss_seidel is: ' + 'x=' + str(x_result))
# print('The computed b is: ' + str(np.dot(A, x_result)))
# print('The real b is: ' + str(b))
