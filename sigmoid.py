import numpy as np

def sigmoid(x):
    """
    Computes sigmoid of x.

    Arguments:
    x -- Could be either a real number, a vector or a matrix.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

y = np.array([1, 2, 3])
print(sigmoid(y))

# np.exp(x) will apply the exponential function to every elelment of x.
# Thus we will end up with sigmoid s who has same size with x.
# The data structures we use in numpy to represent (vectors,matrices,...) is called numpy arrays.
