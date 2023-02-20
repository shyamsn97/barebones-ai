import matplotlib.pyplot as plt
import numpy as np

from barebones_ai import utils


class PCA:
    """
    Dimensionality reduction using the diagonalization of a covariance matrix
    Can specify whether to use calculate the covariance matrix of the rows or columns of a given data matrix
    Parameters:
        X: numpy array() data matrix
        column: boolean determines whether to generate covariance matrix from columns or rows
        eigenvalues and eigenvectors: numpy array() eigenvalues and eigenvectors of the covariance matrix
        proportion_variance: numpy array() variance explained by PCs
        cumulative_var: numpy array() cumulative variance explained by PCs
    """

    def __init__(self, X, column=True):
        # PCA uses the rows of X or the columns to construct the cov matrix
        self.column = column
        self.X = X
        if column:
            self.mumat = X.mean(axis=0)
            self.cov = utils.compute_covariance(X)
        else:
            self.mumat = X.mean(axis=1)
            self.cov = utils.compute_covariance(X, False)
        self.X_shifted = self.X - self.mumat
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov)
        self.eigenvectors = self.eigenvectors.astype(float).real
        self.eigenvalues = np.sort(self.eigenvalues)[::-1]
        self.proportion_variance = (
            (self.eigenvalues / float(sum(self.eigenvalues))).astype(float)
        ).real
        self.cumulative_var = np.cumsum(self.proportion_variance)

    def rank(self, n):
        # this approximates the original matrix by using n eigenvectors corresponding to the biggest eigenvalues
        eigenvalues = self.eigenvalues
        eigenvectors = self.eigenvectors
        indices = eigenvalues.argsort()[::-1][:n]
        Q = eigenvectors[:, indices]
        if self.column is False:
            return Q.dot(Q.T.dot(self.X_shifted)).astype(float) + self.mumat
        else:
            return self.X_shifted.dot(Q).dot(Q.T).astype(float) + self.mumat

    def project(self, n):
        # this projects the data onto a lower dimension
        eigenvalues = self.eigenvalues
        eigenvectors = self.eigenvectors
        indices = eigenvalues.argsort()[::-1][:n]
        Q = eigenvectors[:, indices]

        return self.X.dot(Q).astype(float)


# plotting vals
def plot2D(X, y, classes=False, title="Plotted Vals"):

    fig, ax = plt.subplots()
    if len(X.shape) >= 2:
        pca = PCA(X)
        if classes:
            X = pca.project(2)
            for class_val in np.unique(y):
                i = np.where(y == class_val)[0]
                ax.scatter(x=X[i, 0], y=X[i, 1], label=class_val)
                plt.ylabel("PC 2")
                plt.xlabel("PC 1")
        else:
            X = pca.project(1)
            ax.scatter(X, y)
            plt.ylabel("Y")
            plt.xlabel("X")
    else:
        if classes:
            for class_val in np.unique(y):
                i = np.where(y == class_val)[0]
                ax.scatter(x=X, y=y, label=class_val)
                plt.ylabel("PC 2")
                plt.xlabel("PC 1")
        else:
            ax.scatter(x=X, y=y)
            plt.ylabel("Y")
            plt.xlabel("X")

    ax.legend()
    plt.title(title)
    plt.figure(figsize=(20, 20))
    plt.show()
