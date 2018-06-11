import numpy as np
import sys
from tqdm import tqdm
from nn_optimization_methods import SGD
from layers.Dense import Dense
from layers.Input import Input
from layers.Softmax import Softmax
sys.path.append('../tools/')
import tools

class DNN():
    """
    Deep Neural Network Template for regression and classification tasks
    Parameters:
        X: numpy array() data matrix
        y: numpy array() response variables, must be numeric
        output: numpy array() outputs
        outlayer: output layer
        head: input layer
        loss: loss function str
    
    Use like sequential model from Keras:
        Ex: add a dense layer with 200 neurons and a sigmoid activation function:
                dnn.add("dense","sigmoid",200)

        For classification tasks, the response/class values should be reshaped as a one hot matrix.
    """
    def __init__(self,X,y):
            
        self.X = X
        self.y = y
        self.output = 0
        self.outlayer = Input(self.X)
        self.head = self.outlayer
        self.loss = "mse"
    
    def add(self,layertype,neurons,activation="sigmoid"):
        
        if layertype == "dense":
                layer = Dense(self.outlayer,neurons,activation)
                self.outlayer.setNext(layer)
                self.outlayer = layer
        elif layertype == "softmax":
            layer = Softmax(self.outlayer,neurons)
            self.outlayer.setNext(layer)
            self.outlayer = layer
            
    def forward(self,inputs=None):
        
        if np.all(inputs != None):
            
            self.head.update(inputs)
            
        layer = self.head
        
        while np.all(layer.next != None):
            
            layer.forward()
            layer = layer.getNext()
        
        layer.forward()
        return layer.output
        
    def backward_pass(self,predictions,y):
        
        cache = 0
        gradients = []
                    
        layer = self.outlayer
        cache = np.array((predictions - y))
        w_derivative = layer.getPrev().dot(cache*(layer.deriv),True)
        b_derivative = cache*(layer.deriv)
        gradients.append([w_derivative,b_derivative])
        layer = layer.getPrev()
        while np.all(layer.getPrev() != None):
            cache = (cache.dot(layer.getNext().getWeights()[0].T))*layer.deriv
            w_derivative = layer.getPrev().dot(cache,True)
            b_derivative = cache
            gradients.append([w_derivative,b_derivative])
            layer = layer.getPrev()

        return gradients[::-1]
        
    def train(self,X,y,optimizer=SGD,lr=0.0001,epochs=100,batch_size=1,loss="mse"):
        self.loss = loss
        optimizer(self,X,y,learning_rate=lr,epochs=epochs,batch_size=batch_size,loss=loss)
        
    def predict(self,X):
        if self.loss == "cross_entropy":
            return np.argmax(self.forward(X),axis=1)
        return self.forward(X)
        
    def __str__(self):
        
        string = "----------------------------- Model -----------------------------" + '\n'
        layer = self.head
        
        while np.all(layer != None):
            string += layer.__str__()
            layer = layer.getNext()
            
        return string
    
    
    