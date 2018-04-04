import numpy as np
import sys
sys.path.append('../tools')
import tools

class LDA():
    """
    Linear Discriminant Analysis Classifier using Gaussian assumptions
    Parameters:
    numpy array X: data matrix, must have shape of length two (for vectors, reshape with column = 1)
    numpy array y: class labels, must be numeric
    """
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.priors = self.generate_priors()
        self.params = {"mu":[],"sigma":[]}
        self.predict = None
        self.initialize()
        
    def initialize(self):
        X = self.X
        self.generate_mu()
        self.generate_sigma()
        if X.shape[1] == 1:
            self.predict = self.oned
        else:
            self.predict = self.multd
    
    def generate_priors(self):
        return np.log((np.unique(self.y,return_counts=True)[1]/self.y.shape[0]).reshape(1,np.unique(self.y).shape[0]))
    
    def generate_mu(self):
        self.params["mu"] = []
        X = self.X
        y = self.y
        indices = np.arange(y.shape[0])
        uniquevals = np.unique(y)
        if X.shape[1] == 1:
            for i in uniquevals:
                find = np.where(y == i)[0]
                self.params["mu"].append(np.mean(X[find]))
            self.params["mu"] = np.array(self.params["mu"]).reshape(1,uniquevals.shape[0])
        else:
            for i in uniquevals:
                find = np.where(y == i)[0]
                self.params["mu"].append(np.mean(X[find],axis=0))
            
    def generate_sigma(self):
        self.params["sigma"] = []
        X = self.X
        y = self.y
        indices = np.arange(y.shape[0])
        uniquevals = np.unique(y)
        if X.shape[1] == 1:
            var = 0
            for i in uniquevals:
                find = np.where(y == i)[0]
                var += np.var(X[find])
            self.params["sigma"] = var/(X.shape[0]-uniquevals.shape[0])
        else:
            self.params["sigma"] = tools.compute_covariance(X)
                
    def oned(self,predictors):
        y = self.y
        ones = np.ones(predictors.shape[0]).reshape(predictors.shape[0],1)
        return np.argmax(predictors *ones.dot(self.params["mu"]/self.params["sigma"]) - 
                         ones.dot((self.params["mu"])**2/(2*self.params["sigma"])) + 
                         ones.dot(self.priors),axis=1)
    
    def multd(self,predictors):
        y = self.y
        mu = self.params["mu"]
        sigma = self.params["sigma"]
        ones = np.ones(predictors.shape[0]).reshape(predictors.shape[0],1)
        probs = ones
        priors = self.priors[0]
        for i in np.unique(y):
            mureshaped = mu[i].reshape(1,predictors.shape[1])
            first = predictors.dot(np.linalg.inv(sigma)).dot(mureshaped.T) 
            second = (ones.dot((mureshaped/2).dot(np.linalg.inv(sigma)).dot(mureshaped.T)))
            prob = first - second + ones*priors[i]
            probs = np.column_stack((probs,prob))
        probs = probs[:,1:]
        return np.argmax(probs,axis=1)