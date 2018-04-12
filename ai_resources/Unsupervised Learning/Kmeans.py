import numpy as np
import sys
sys.path.append('../tools')
import tools
class Kmeans():
    
    def __init__(self,X):
        self.X = standardize(X)
        self.center_assignments = {}
        self.centers = []
        
    def calculate_centers(self):
        self.centers = np.array(list(map(lambda x : np.mean(X[x,:],axis=0),self.center_assignments.values())))
        
    def predict(self,k,seed,exit=0.01):
        
        X = self.X
        self.centers = np.random.uniform(0,1,size=(k,X.shape[1]))
        oldcenters = self.centers*2
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