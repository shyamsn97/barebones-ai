import numpy as np
import sys
sys.path.append('../tools/')
import tools

class Input():
    """
    Input Layer used for the DNN object
    Parameters:
        weights: numpy array() weight matrix
        output: output of the layer
        deriv: derivative of activation
        shape: output shape
        prevlayer: previous layer
        next: next layer
    """
    def __init__(self,X):
        
        self.weights = X
        self.deriv = 1
        self.output = X
        self.shape = X.shape
        self.prevlayer = None
        self.next = None
        
    def getWeights(self):
        
        return self.weights
       
    def setNext(self,layer):
        
        self.next = layer
        
    def getNext(self):
        
        return self.next
    
    def getPrev(self):
        
        return self.prevlayer
    
    def dot(self,X,t=False):
        
        if t == True:
            return self.output.T.dot(X)
        return self.output.dot(X)
    
    def foward(self):
        self.output = self.weights
    
    def predict(self):
        return self.weights
    
    def update(self,X):
        self.weights = X
    
    def __repr__(self):
        
        string = "Input: " + "shape: " + str((None,self.weights.shape[1]))
        string = string + '\n' + "-----------------------------------------------------------------" + '\n'
        return string  
    
    def __str__(self):
        
        string = "Input: " + "shape: " + str((None,self.weights.shape[1]))
        string = string + '\n' + "-----------------------------------------------------------------" + '\n'
        return string  