import numpy as np

class FuzzyKmeans():
    """
    Fuzzy Kmeans calculates cluster probabilities in regards to euclidian distance
    Equivalent to vanilla Kmeans if we assign a given point to the cluster with the highest fuzzy probability
    Parameters:
        X: numpy array() data matrix
        U: cluster probabilities
        centers: center values
        m: numeric, the degree of uncertainty, (fuzziness of cluster)
    """
    def __init__(self,X,m=2):
        self.X = X
        self.m = m
        self.U = None
        self.centers = None

    def calculate_centers(self,U,X):
        """
        Recalculates centers using a linear combination of fuzzy probs and X values
        Normalized by the sum of all the fuzzy probs.
        """
        m = self.m
        ones = np.ones(X.shape[0]).reshape((X.shape[0],1))
        denom = (1/(U**m).dot(ones)).reshape(U.shape[0],)
        diagonal = np.diag(denom)
        centers = diagonal.dot((U**m).dot(X))
        return centers
        
    def calculate_fuzzy(self,X,centers):
        m = self.m
        m = 2/(m-1)
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
        
    def fit(self,k,exit=0.01,seed=1,maxiterations=100,argmax=False): 
        """
        Main clustering function
            k is the number of clusters
            exit is the exit criteria 
            set argmax = True for normal K-means
        """
        X = self.X
        np.random.seed(seed)
        U = np.random.uniform(0,1,size=(k,X.shape[0])) #initialize cluster probabilities
        centers = self.calculate_centers(U,X)
        newcenters = 2*centers
        count = 0
        while np.linalg.norm((centers - newcenters),2) >= exit and count <= maxiterations:
            newcenters = centers
            U = self.calculate_fuzzy(X,centers)
            centers = self.calculate_centers(U,X)
            count += 1
        self.U = U
        self.centers = centers

    def predict(self,X,argmax=False):

        U = self.calculate_fuzzy(X,self.centers)
        if argmax:
            return (np.argmax(U,axis=0).T).reshape(X.shape[0],)
        return U.T