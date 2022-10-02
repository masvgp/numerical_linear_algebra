# Jacobi iterations method for solving Ax=b
import numpy as np


def jacobi(A, b, x_start, epsilon=1e-10, max_iterations=500):
  """
  Jacobi iteration method for solving the system of equations Ax=b.
  The Matrix equation is x_next = D^{-1} * (b - M * x_previous) where D^{-1} is the inverse of the diagonal elements of A, M is the remaining elements of A after the diagonals are subtracted, and b is b.
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
  """
  # select out diagonal elements of A
  D = np.eye(A.shape[0], A.shape[1])
  D = D * A.diagonal()
  D_inv = np.diag(1 / np.diag(D))
  M = A - D
  x = x_start
  for i in range(max_iterations):
    x_next = np.dot(D_inv, b - np.dot(M, x))

    if np.linalg.norm(x - x_next) < epsilon:
      return x
    x = x_next

  return x


# Test the function using data from Wikipedia
# This example converges in 69 iterations.
# problem data
# A = np.array([
#     [5, 2, 1, 1],
#     [2, 6, 2, 1],
#     [1, 2, 7, 1],
#     [1, 1, 2, 8]
# ])
# b = np.array([29, 31, 26, 19])

# # you can choose any starting vector
# x_start = np.zeros(len(b))
# x = jacobi(A, b, x_start)

# print("x:", x)
# print("computed b:", np.dot(A, x))
# print("real b:", b)
