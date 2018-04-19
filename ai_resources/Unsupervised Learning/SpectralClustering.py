import numpy as np
sys.path.append('../tools')
import tools
sys.path.append('../Unsupervised Learning/')
from FuzzyKmeans import FuzzyKmeans

class SpectralClustering():
    """
        Clustering algorithm that uses the eigenvectors of the normalized laplacian
        Parameters:
            X: numpy array() data matrix
    """
    def __init__(self,X):
        
        self.X = X
        
    def generate_normalized_laplacian(self,distance_measurement='l2'):
        
        X = self.X
        similarity = tools.generate_similarity_matrix(X)
        distances_inv = np.diag(1/np.sum(similarity,axis=1))
        laplacian = np.identity(X.shape[0]) - distances_inv.dot(similarity)
        return laplacian
    
    def predict(self,k,distance_measure='l2',clustering_algo=FuzzyKmeans):
        
        laplacian = self.generate_normalized_laplacian(distance_measure)
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        indices = eigenvalues.argsort()[:k]
        eigenvectors = eigenvectors[:,indices]
        clustering = clustering_algo(eigenvectors)
        return clustering.predict(k,argmax=True)
    