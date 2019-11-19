import math

def basic_sigmoid(x):
    """
    Computes sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s

print(basic_sigmoid(3))

# Actually, we rarely use "math" library in deep learning because the inputs of the functions are real numbers.
# In deep learning we mostly use matrices and vectors. This is why "numpy" library is more useful.
