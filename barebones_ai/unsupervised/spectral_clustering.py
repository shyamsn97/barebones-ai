import numpy as np

from barebones_ai import utils
from barebones_ai.unsupervised.kmeans import Kmeans


class SpectralClustering:
    """
    Clustering algorithm that uses the eigenvectors of the normalized laplacian
    Parameters:
        X: numpy array() data matrix
    """

    def __init__(self, X):
        self.X = X

    def generate_normalized_laplacian(self, distances="l2"):

        X = utils.generate_similarity_matrix(self.X, distances)
        distances = np.diag(np.sum(X, axis=1))
        L = distances - X
        distances_inv = np.diag(1 / np.sum(X, axis=1))
        laplacian = distances_inv.dot(L)
        return laplacian

    def predict(self, k, distances="l2", clustering_algo=Kmeans):

        laplacian = self.generate_normalized_laplacian(distances)
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        indices = eigenvalues.argsort()[:k]
        eigenvectors = eigenvectors[:, indices]
        clustering = clustering_algo(eigenvectors)
        clustering.fit(k)
        return clustering.predict(eigenvectors)
