import numpy as np
sys.path.append('../tools')
import tools
sys.path.append('../Unsupervised_Learning/')
from FuzzyKmeans import FuzzyKmeans

class SpectralClustering():
    """
        Clustering algorithm that uses the eigenvectors of the normalized laplacian
        Parameters:
            X: numpy array() data matrix
    """
      def __init__(self,X):
        
        self.X = X
        
    def generate_normalized_laplacian(self,distances='l2'):
        
        X = tools.generate_similarity_matrix(self.X,distances)
        distances = np.diag(np.sum(X,axis=1))
        l = distances - X
        distances_inv = np.diag(1/np.sum(X,axis=1))
        laplacian = distances_inv.dot(l)
        return laplacian
    
    def predict(self,k,distances='l2',clustering_algo=FuzzyKmeans):
        
        laplacian = self.generate_normalized_laplacian(distances)
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        indices = eigenvalues.argsort()[:k]
        eigenvectors = eigenvectors[:,indices]
        clustering = clustering_algo(eigenvectors)
        return clustering.predict(k,seed=0)
    
    