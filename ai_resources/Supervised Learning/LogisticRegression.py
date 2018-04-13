import numpy as np
import sys
sys.path.append('../tools')
import tools
import iterative_methods

class LogisticRegression():
    
    """
    Logistic Regression class for binary classification
    Parameters:
        X: numpy array() data matrix, must have shape of length two (for vectors, reshape with column = 1)
        y: numpy array() class labels, must be numeric
        weights: numpy array() weights for prediction
    """

    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        self.weights = np.random.uniform(10,size=X.shape[1])
        
    def gradient_func(self,parameters,X,y):
        
        h = self.predict(X,parameters)
        return (X.T.dot(h-y))
    
    def predict(self,X,parameters = 0):
        
        if np.all(parameters == 0):
            parameters = self.weights
            predictions = tools.sigmoid(X.dot(parameters)).astype(float)
            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            return predictions
        return tools.sigmoid(X.dot(parameters)).astype(float)
    
    def train(self):
        
        self.weights = iterative_methods.Batch_Gradient_Descent(self.X,self.y,self.weights,self.gradient_func,self.predict)
        
        
