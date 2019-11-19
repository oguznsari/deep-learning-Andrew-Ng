import numpy as np

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix # XXX:

    Argument:
    x -- A numpy matrix of shape(n,m)

    Returns:
    x -- The normalized(by row) numpy matrix. You are allowed to modify x.
    """

    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm

    return x

x = np.array([[0, 3, 4],
              [1, 6, 4]])

print("normalizeRows(x) = " + str(normalizeRows(x)))

# It often leads to better performance because gradient descent converges faster after normalization
# Normalization = diving each vector of x by its norm
