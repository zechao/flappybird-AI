import numpy as np


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def tanh(X):
    return np.tanh(X)


def relu(X):
    return np.maximum(0, X)


def dsigmoid(X):
    return np.multiply(X, (1 - X))


def dtanh(X):
    return (1 - np.square(X))


def drelu(X):
    return 1.0 * (X > 0)
