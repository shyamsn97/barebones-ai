
import numpy as np

class FuzzyKmeans():
    
    """
    Fuzzy Kmeans calculates cluster probabilities in regards to euclidian distance
    Equivalent to vanilla Kmeans if we assign a given point to the cluster with the highest fuzzy probability
    Parameters:
        X: numpy array() data matrix
    """
    def __init__(self,X):
        self.X = X
        
    def calculate_centers(self,U,m):
        """
        Recalculates centers using a linear combination of fuzzy probs and X values
        Normalized by the sum of all the fuzzy probs.
        """
        X = self.X
        ones = np.ones(X.shape[0]).reshape((X.shape[0],1))
        denom = (1/(U**m).dot(ones)).reshape(U.shape[0],)
        diagonal = np.diag(denom)
        centers = diagonal.dot((U**m).dot(X))
        return centers
        
    def calculate_fuzzy(self,centers,m):
        m = 2/(m-1)
        X = self.X
        us = np.ones(shape=(centers.shape[0],1))
        ones = np.ones(shape=(1,centers.shape[0]))
        for i in range(X.shape[0]):
            row = X[i].reshape(X.shape[1],1)
            row = row.dot(ones).T - centers
            norms = np.linalg.norm(row,2,axis=1).reshape(centers.shape[0],1)
            norms = (norms.dot(1/norms.T))**m
            sums = 1/np.sum(norms,axis=1).reshape(norms.shape[0],1)
            us = np.column_stack((us,sums))

        return us[:,1:]
        
    def cluster(self,m,k,exit,seed,maxiterations=100,argmax=False): 
        """
        Main clustering function
        m is the degree of uncertainty, (fuzziness of cluster)
        k is the number of clusters
        exit is the exit criteria 
        set argmax = True for normal K-means
        """
        X = self.X
        np.random.seed(seed)
        U = np.random.uniform(0,1,size=(k,X.shape[0])) #initialize cluster probabilities
        centers = self.calculate_centers(U,m)
        newcenters = 2*centers
        count = 0
        while np.linalg.norm((centers - newcenters),2) >= exit and count <= maxiterations:
            newcenters = centers
            U = self.calculate_fuzzy(centers,m)
            centers = self.calculate_centers(U,m)
            count += 1
        if argmax:
            return (np.argmax(U,axis=0).T).reshape(X.shape[0],1)
        return U.T