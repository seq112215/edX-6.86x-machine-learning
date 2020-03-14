import numpy as np


np.random.seed(0)


def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    # Your code here

    A = np.asarray(np.random.rand(n, 1))

    return A


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    # Your code here

    A = np.random.rand(h, w)
    B = np.random.rand(h, w)

    s = np.add(A, B)

    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    # Your code here

    s = np.linalg.norm(np.add(A, B))

    return s
