import numpy as np
import pandas as pd
import sys
sys.path.append('../tools')
import tools

class NaiveBayes():
    """
    Naive bayes classifier that utilizes the unique property of conditional independence to approximate probabilities
    Parameters:
        X: numpy array() with n x p columns, p generally < n
        y: numpy array() of labels(classes)
        self.bins: quantile bins of data matrix X, used to bin predictions
        binning: set to true if dealing with continuous variables, where values are binned into four quantiles
        conditionals: dictionary with numpy arrays() of conditional bayesian probs
        priors: dictionary with numpy arrays() of prior probabilities      
        prior_indices: lookup table for the indices where y vals are located
    """
    def __init__(self,X,y,binning=False):

        self.X = X.copy()
        self.y = y.copy()
        self.bins = {}
        if binning == True:
            self.binvals(self.X)
        self.binning = binning
        self.conditionals = {}
        self.priors = {}
        self.prior_indices = {}
        self.get_priors()
    
    def binvals(self,values):
        for i in range(values.shape[1]):
            self.bins[i] = pd.qcut(values[:,i],4,retbins=True)[1]
            values[:,i] = pd.cut(values[:,i], self.bins[i], labels=False, include_lowest=True)
            
    def get_priors(self):
        """
        Prior probs for each class y, P(Y=y)
        """
        uniquevals = np.unique(self.y)
        for i in uniquevals:
            indices = np.where(self.y == i)[0]
            self.prior_indices[i] = indices
            self.priors[i] = indices.shape[0]/self.y.shape[0]
            
    def get_conditionals(self,observation,y_val):
        
        indices = self.prior_indices[y_val]
        size = self.X.shape[0]
        prob = 1
        for i in range(observation.shape[1]):
            values = self.X[indices,i]
            values = np.where(values == observation[:,i])[0]
            prob = prob*((values.shape[0]/size))
        return prob
            
    def predict(self,observations):
        """
            observations must be either vectors of shape 1xp or matrices where rows = observations
        """
        if self.binning == True:
            observations = observations.copy()
            for i in range(observations.shape[1]):
                observations[:,i] = pd.cut(observations[:,i], self.bins[i], labels=False, include_lowest=True)
        if observations.shape[1] != self.X.shape[1]:
            print("Error: observation columns not the same rank as data X")
            return None
        
        predictions = []
        columns = self.X.shape[1]
        for obs in range(observations.shape[0]):
            obs_prediction = []
            for key in list(self.priors.keys()):
                val = self.get_conditionals(observations[obs].reshape(1,columns),key)
                obs_prediction.append(val)
            predictions.append(np.argmax(obs_prediction))
        return np.array(prediction).reshape(len(predictions),)
                
                
                
                
            
        
        
    
        