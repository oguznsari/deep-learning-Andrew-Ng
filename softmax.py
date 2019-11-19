import numpy as np

def softmax(x):
    """
    Calculates the softmax for each row of the input x.
    Code should work for a row vector and also for matrices of shape(m,n).

    Argument:
    x -- A numpy matrix of shape(m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape(m,n)
    """
    x_exp = np.exp(x)                                   # Applies exp() element-wise to x.
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)    # Sums each row of x_exp
    s = x_exp / x_sum                                   # Computes softmax(x) by dividing x_exp by x_sum

    return s

x = np.array([[9, 2, 5, 0, 0],
              [7, 5, 0, 0, 0]])

print("softmax(x) = " + str(softmax(x)))

# See that x_sum is of shape(2,1) while x_exp and s are of shape(2,5).  # x_exp / x_sum works due to python "broadcasting"
