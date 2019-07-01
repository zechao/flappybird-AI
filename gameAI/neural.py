import numpy as np


def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

np.random.