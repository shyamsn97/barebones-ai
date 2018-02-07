
import numpy as np
import sys
sys.path.append('../tools')
import tools

class PCA():
    
    def __init__(self, X):     
        self.X = compute_covariance(X)
        
    def rank(self,n): 
        #this approximates the original matrix by using n eigenvectors corresponding to the biggest eigenvalues    
        eigenvalues, eigenvectors = np.linalg.eig(self.X)  
        indices = eigenvalues.argsort()[::-1][:n]  
        Q = eigenvectors[:,indices]
         
        return Q.dot(Q.T.dot(X))

    def project(self,n):
        #this projects the data onto a lower dimension
        eigenvalues, eigenvectors = np.linalg.eig(self.X) 
        indices = eigenvalues.argsort()[::-1][:n]
        Q = eigenvectors[:,indices]
        
        return X.dot(Q)
        
        