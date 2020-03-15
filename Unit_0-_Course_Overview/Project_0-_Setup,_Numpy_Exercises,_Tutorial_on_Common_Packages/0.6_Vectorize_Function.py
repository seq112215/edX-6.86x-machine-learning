import numpy as np


def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    # Your code here

    if x <= y:
        return x*y
    else:
        return x/y


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    # Your code here

    return np.vectorize(scalar_function)(x, y)
