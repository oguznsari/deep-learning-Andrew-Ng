import numpy as np

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- np.dot(y-yhat, y-yhat)
    """

    loss = np.dot(y-yhat, y-yhat)

    return loss

yhat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat, y)))
