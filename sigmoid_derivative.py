import numpy as np

def sigmoid_derivative(x):
    """
    Compute the gradient (also called slope or derivative) of the sigmoid function

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Computed gradient.
    """

    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)

    return ds

x = np.array([1, 2, 3])
print(sigmoid_derivative(x))
