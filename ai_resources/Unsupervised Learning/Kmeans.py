import numpy as np
import sys
sys.path.append('../tools')
import tools
class Kmeans():
    
    """
    Vanilla Kmeans:
    Parameters:
        X: numpy array() data matrix
        center_assignments: dictionary of numpy arrays of indices for data points
        centers: list of numpy arrays for centers of clusters
    """
    def __init__(self,X):
        self.X = X
        self.center_assignments = {}
        self.centers = []
        
    def calculate_centers(self):
        self.centers = np.array(list(map(lambda x : np.mean(X[x,:],axis=0),self.center_assignments.values())))
        
    def predict(self,k,seed,exit=0.001):
        
        X = self.X
        self.centers = np.zeros(shape=(k,X.shape[1]))
        print(self.centers.shape)
        for i in range(k):
            for j in range(X.shape[1]):
                minx = np.min(X[:,j])
                maxx = np.max(X[:,j])
                self.centers[i,j] = np.random.uniform(minx,maxx)
        oldcenters = self.centers + 4
        while np.all(np.linalg.norm((self.centers - oldcenters),2) >= exit):
            oldcenters = self.centers
            for i in range(k):
                self.center_assignments[i] = []
            for i in range(X.shape[0]):
                ones = np.ones(oldcenters.shape[0]).reshape(oldcenters.shape[0],1)
                sample = ones.dot(X[i,:].reshape(1,X.shape[1]))
                closest = np.argmin(np.linalg.norm(oldcenters - sample,2,axis=1))
                self.center_assignments[closest].append(i)
                self.calculate_centers()
        end = np.zeros(X.shape[0])
        for i in range(k):
            end[self.center_assignments[i]] = i
        
        return end