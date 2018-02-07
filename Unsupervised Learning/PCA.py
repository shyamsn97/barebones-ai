
import numpy as np

def compute_covariance(X, correlation=False):

    me = np.mean(X,axis=0)
    meanmat = np.outer(me,np.ones(X.shape[0])).T
    X_cent = X - meanmat
    X_cent = X_cent.dot(X_cent.T)/(X_cent.shape[0] - 1)

    if correlation == True:

        stdev = np.diag(1/np.sqrt(np.diag(X_cent)))
        X_cent = stdev.dot(X_cent).dot(stdev)

    return X_cent


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
        
        