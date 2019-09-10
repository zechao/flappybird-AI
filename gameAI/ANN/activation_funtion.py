import numpy as np
import time

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


if __name__ == '__main__':
    start=time.time()
    result =0
    for x in range(100000):
        result = sigmoid(x)
    print("total time for sigmoid:",time.time()-start)

    start = time.time()
    result = 0
    for x in range(100000):
        result = relu(x)
    print("total time for relu:", time.time() - start)
    print(sigmoid(-20))