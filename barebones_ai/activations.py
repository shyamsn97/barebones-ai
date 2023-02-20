import numpy as np


# activations
def sigmoid(x, derivative=False):
    """
    sigmoid function, set derivative = true to get the derivative
    """
    if derivative:
        return 1 / (1 + np.e ** -(x * 1.0)) * (1 - (1 / (1 + np.e ** -(x * 1.0))))
    else:
        return 1 / (1 + np.e ** -(x * 1.0))


def softmax(x, derivative=False):
    """
    stable softmax function, set derivative = true to get the derivative
    """
    vecs = np.exp(x - np.max(x))
    if derivative:
        if len(x.shape) > 1:
            s = vecs / (np.sum(vecs, axis=1).reshape(x.shape[0], 1))
            return s * (1 - s)
        else:
            s = vecs / (np.sum(vecs))
            return s * (1 - s)
    else:
        if len(x.shape) > 1:
            return vecs / (np.sum(vecs, axis=1).reshape(x.shape[0], 1))
        else:
            return vecs / (np.sum(vecs))


def relu(x, derivative=False):
    """
    relu function, set derivative = true to get the derivative
    """
    rel = np.maximum(x, 0)
    if derivative:
        rel[x < 0] = 0
        rel[x >= 0] = 1
    return rel
