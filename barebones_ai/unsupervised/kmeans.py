import numpy as np

from barebones_ai import utils


class Kmeans:
    """
    Kmeans / Kmedians:
    Parameters:
        X: numpy array() data matrix
        center_assignments: dictionary of numpy arrays of indices for data points
        centers: list of numpy arrays for centers of clusters
    """

    def __init__(self, X):
        self.X = X
        self.center_assignments = {}
        self.centers = []

    def calculate_centers(self, dist=1):
        X = self.X
        if dist == 1:
            self.centers = np.array(
                list(
                    map(
                        lambda x: np.mean(X[x, :], axis=0),
                        self.center_assignments.values(),
                    )
                )
            )
        elif dist == 2:
            self.centers = np.array(
                list(
                    map(
                        lambda x: np.median(X[x, :], axis=0),
                        self.center_assignments.values(),
                    )
                )
            )

    def fit(self, k, seed=0, exit=0.01, dist_type="mean", maxiterations=100):
        self.center_assignments = {}
        self.centers = []
        X = self.X
        self.centers = np.zeros(shape=(k, X.shape[1]))
        for i in range(k):
            for j in range(X.shape[1]):
                minx = np.min(X[:, j])
                maxx = np.max(X[:, j])
                self.centers[i, j] = np.random.uniform(minx, maxx)
        oldcenters = self.centers + 4
        count = 0
        while np.all(np.linalg.norm((self.centers - oldcenters), 2) >= exit):
            count += 1
            if count == maxiterations:
                break
            oldcenters = self.centers
            for i in range(k):
                self.center_assignments[i] = []
            for i in range(X.shape[0]):
                ones = np.ones(oldcenters.shape[0]).reshape(oldcenters.shape[0], 1)
                sample = ones.dot(X[i, :].reshape(1, X.shape[1]))
                closest = np.argmin(np.linalg.norm(oldcenters - sample, 2, axis=1))
                self.center_assignments[closest].append(i)
                if dist_type == "mean":
                    self.calculate_centers(1)
                elif dist_type == "median":
                    self.calculate_centers(2)
        self.centers = np.array(self.centers)

    def predict(self, X):
        clusters = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            clusters[i] = np.argmin(utils.l2distance(X[i], self.centers))
        return clusters
