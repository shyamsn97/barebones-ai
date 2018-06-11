import numpy as np
import sys
sys.path.append('../tools/')
import tools

class Softmax():
    """
    Softmax Layer used for the DNN object
    Parameters:
        weights: numpy array() weight matrix
        output: output of the layer
        deriv: derivative of activation
        shape: output shape
        prevlayer: previous layer
        next: next layer
    """
    
    def __init__(self,prevlayer,outputdims):
        
        self.weights = [np.random.uniform(-1,1,size=(prevlayer.shape[1],outputdims)),np.random.uniform(-1,1,size=(1,outputdims))]
        self.output = 0
        self.deriv = 1
        self.activation_name = "softmax"
        self.activation = tools.softmax
        self.print_shape = (None,outputdims)
        self.shape = (prevlayer.shape[1],outputdims)
        self.prevlayer = prevlayer
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
        
    def forward(self):
        
        layer_mul = self.prevlayer.dot(self.weights[0]) + self.weights[1]
        self.output = self.activation(layer_mul)
        self.deriv = self.activation(layer_mul,True)
    
    def predict(self,X):
        
        return self.activation(X.dot(self.weights[0]) + self.weights[1])
        
    def update(self,X):

        self.weights = X
        
    def __repr__(self):
        
        string = "Softmax: " + "activation: " + self.activation_name +  ", weight shape: " + str(self.weights[0].shape) + ", output shape: " + str(self.print_shape)
        string = string + ", parameters: " + str(self.weights[0].shape[0]*self.weights[0].shape[1])
        string = string + '\n' + "-----------------------------------------------------------------" + '\n'
        return string  
    
    def __str__(self):
        
        string = "Softmax: " + "activation: " + self.activation_name +  ", weight shape: " + str(self.weights[0].shape) + ", output shape: " + str(self.print_shape)
        string = string + ", parameters: " + str(self.weights[0].shape[0]*self.weights[0].shape[1])
        string = string + '\n' + "-----------------------------------------------------------------" + '\n'
        return string   
         
