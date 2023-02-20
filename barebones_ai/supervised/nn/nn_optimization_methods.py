import numpy as np
from tqdm import tqdm

from barebones_ai import utils


def SGD(dnn, X, y, learning_rate=0.0001, epochs=100, batch_size=1, loss="mse"):
    """
    Stochastic Gradient Descent for Neural Networks
    """
    bar = tqdm(np.arange(epochs))
    for i in bar:

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        sample = 0
        count = 0

        while sample < indices.shape[0]:

            batch_X = X[indices[sample : (sample + batch_size)]]
            batch_y = y[indices[sample : (sample + batch_size)]]
            batch_h = dnn.forward(batch_X)
            sample += batch_size
            gradients = dnn.backward_pass(batch_h, batch_y)
            layer = dnn.head.getNext()
            count += 1
            j = 0

            while np.all(layer is not None):

                new_weights = layer.getWeights()
                new_weights[0] -= learning_rate * gradients[j][0]
                new_weights[1] -= learning_rate * gradients[j][1]
                layer.update(new_weights)
                layer = layer.getNext()
                j += 1

        h = dnn.forward(X)
        if loss == "mse":
            bar.set_description("MSE %s" % str(utils.mean_squared_error(h, y)))
        elif loss == "cross_entropy":
            bar.set_description("Cross Entropy %s" % str(utils.cross_entropy(h, y)))
