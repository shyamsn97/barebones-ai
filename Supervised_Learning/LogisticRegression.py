import numpy as np
import sys
sys.path.append('../tools')
import tools
import iterative_methods

class LogisticRegression():
    """
    Logistic Regression class for binary classification
    Parameters:
        X: numpy array() data matrix
        y: numpy array() class labels, must be numeric
        weights: numpy array() weights for prediction
    """
    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        self.weights = np.random.uniform(10,size=X.shape[1])
        
    def gradient_func(self,parameters,X,y):
        
        h = self.predict(X,parameters)
        return (1/h.shape[0])*(X.T.dot(h-y))
    
    def predict(self,X,parameters=None):
        
        if np.all(parameters == None):
            parameters = self.weights
            predictions = tools.sigmoid(X.dot(parameters)).astype(float)
            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            return predictions
        return tools.sigmoid(X.dot(parameters)).astype(float)
    
    def fit(self,batch_size=1,epochs=100,learning_rate=0.001):
        
        self.weights = iterative_methods.Mini_Batch_Gradient_Descent(self.X,self.y,
            self.weights,self.gradient_func,self.predict,epochs=epochs,batch_size=batch_size,loss="cross_entropy",learning_rate=learning_rate)
        
        print("Train Accuracy: %s" % str(tools.calc_accuracy(self.predict(self.X),self.y)))
