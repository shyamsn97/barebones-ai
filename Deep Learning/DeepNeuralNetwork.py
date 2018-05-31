import numpy as np
import sys
from tqdm import tqdm
from nn_optimization_methods import SGD
from layers.Dense import Dense
from layers.Input import Input
sys.path.append('../tools/')
import tools

class DNN():
    """
    Deep Neural Network Template for regression and classification tasks
    Parameters:
        X: numpy array() data matrix
        y: numpy array() response variables, must be numeric
        output: numpy array() outputs
        outlayer = output layer
        head = input layer
    
    Use like sequential model from Keras:
        Ex: add a dense layer with 200 neurons and a sigmoid activation function:
                dnn.add("dense","sigmoid",20)
    """
    def __init__(self,X,y):
            
        self.X = X
        self.y = y
        self.output = 0
        self.outlayer = Input(self.X)
        self.head = self.outlayer
    
    def add(self,layertype,activation,neurons):
        
        if layertype == "dense":
                layer = Dense(self.outlayer,neurons,activation)
                self.outlayer.setNext(layer)
                self.outlayer = layer
            
    def foward(self,inputs=None):
        
        if np.all(inputs != None):
            
            self.head.update(inputs)
            
        layer = self.head
        
        while np.all(layer.next != None):
            
            layer.foward()
            layer = layer.getNext()
        
        layer.foward()
        return layer.output
        
    def backward_pass(self,predictions,y,loss="MSE"):
        
        cache = 0
        gradients = []
        
        if loss == "MSE":
            
            layer = self.outlayer
            cache = np.array(np.sum(predictions - y))
            derivative = layer.getPrev().dot(cache*(layer.deriv),True)
            gradients.append(derivative)
            layer = layer.getPrev()
            
            while np.all(layer.getPrev() != None):
                cache = (cache.dot(layer.getNext().getWeights().T))*layer.deriv
                gradients.append(layer.getPrev().dot(cache,True))
                layer = layer.getPrev()
            
            return gradients[::-1]
        
    def train(self,X,y,optimizer=SGD):
        
        optimizer(self,X,y)
        
    def predict(self,X):
        
        return self.foward(X)
        
    def __str__(self):
        
        string = "----------------------------- Model -----------------------------" + '\n'
        layer = self.head
        
        while np.all(layer != None):
            string += layer.__str__()
            layer = layer.getNext()
            
        return string
    
    