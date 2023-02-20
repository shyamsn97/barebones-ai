import numpy as np


class LinearRegression:
    """
    Linear Regression class, with kfold cross validation
    Parameters:
        X: numpy array() data matrix, must have shape of length two (for vectors, reshape with column = 1)
        y: numpy array() class labels, must be numeric
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.weights = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    def predict(self, X):
        return X.dot(self.weights)

    def k_fold_cross_val(self, k, seed):
        if k <= 1:
            print("Pick a k greater than 1")
            return None

        ones = np.ones(self.X.shape[0])
        X = np.column_stack((ones, self.X)).astype(int)
        y = self.y
        if seed != -1:
            np.random.seed(seed)
        else:
            pass
        fold_size = int(X.shape[0] / k)
        xfolds = []
        yfolds = []
        indexlist = np.arange(X.shape[0])
        newy = np.array(y)
        for i in range(k):
            indices = np.random.choice(indexlist, fold_size, replace=False)
            indexlist = np.delete(indexlist, indices)
            xfolds.append(X[indices, :])
            yfolds.append(newy[indices])
        mses = []
        mspes = []
        for i in range(k):
            leaveoutx = np.array(xfolds[i])
            leaveouty = np.array(yfolds[i])
            leavein = [index for index in range(k) if index != i]
            leaveinx = np.array([xfolds[i] for i in leavein])
            leaveinx = leaveinx.reshape(
                (leaveinx.shape[0] * leaveinx.shape[1], leaveinx.shape[3])
            )
            leaveiny = np.array([yfolds[i] for i in leavein])
            leaveiny = leaveiny.reshape(leaveiny.shape[0] * leaveiny.shape[1])
            betaestimate = np.linalg.inv(leaveinx.T.dot(leaveinx)).dot(
                leaveinx.T.dot(leaveiny)
            )
            trainest = leaveinx.dot(betaestimate)
            testest = leaveoutx.dot(betaestimate)
            MSE = (1 / leaveinx.shape[0]) * (
                (leaveiny - trainest).T.dot((leaveiny - trainest))
            )
            MSPE = (1 / leaveoutx.shape[0]) * (
                (leaveouty - testest).T.dot((leaveouty - testest))
            )
            mses.append(MSE)
            mspes.append(MSPE)

        avgmse = sum(mses) / k
        avgmspe = sum(mspes) / k
        print("Average MSE: " + str(avgmse))
        print("Average MSPE: " + str(avgmspe[0, 0]))

        return avgmse, avgmspe[0, 0]
