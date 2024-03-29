import numpy as np

from barebones_ai import utils


class KNN:
    """
    K nearest neighbor classifier. Assign a given vector to a class based on l2 distance
    Parameters:
        X: numpy array() data matrix
        y: numpy array() labels/classes
    """

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def predict(self, k, pred):
        classes = []
        for i in range(pred.shape[0]):
            predic = pred[i, :]
            predic = predic.reshape(1, predic.shape[0])
            dist = utils.l2distance(predic, self.X)
            indices = dist.argsort()[:k]
            classcounts = self.y[indices]
            vals, counts = np.unique(classcounts, return_counts=True)
            ind = np.argmax(counts)
            classpick = vals[ind]
            classes.append(classpick)

        return np.array(classes)
