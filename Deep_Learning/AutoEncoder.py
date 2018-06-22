import numpy as np
import sys
from nn_optimization_methods import SGD
from layers.Dense import Dense
from layers.Input import Input
from layers.Softmax import Softmax
from DeepNeuralNetwork import DNN
import copy

class AutoEncoder():
    """
    An auto encoder is a semi supervised learning algorithm that attempts to reconstruct input using a smaller feature space
    Parameters:
        X: numpy array(): data matrix
        encoder: DNN to reduce dimensions of matrix
        decoder: DNN to recreate the original data from the encoded data
        full_model: DNN that combines both the encoder and decoder objects, used to train both
    """
    def __init__(self,X):
        
        self.X = X
        self.encoder = None
        self.decoder = None
        self.full_model = DNN()
        self.full_model.add(Input(X))
        self.count = 0
        
    def create_encoder(self,layers=[Dense(32),Dense(512)],encoded_dims=2):
        
        self.count = 0
        for layer in layers:
            self.full_model.add(layer)
            self.count += 1
            
        self.full_model.add(Dense(encoded_dims))
        
    def create_decoder(self,layers=[Dense(32)]):
        
        if len(layers) > 0:
            for layer in layers:
                self.full_model.add(layer)
            
        self.full_model.add(Dense(self.X.shape[-1]))
    
    def finalize_encoder_decoder(self):
        
        count = 0
        layer = self.full_model.head.getNext()
        self.encoder = DNN()
        self.decoder = DNN()
        self.encoder.add(Input(self.X))
        
        while layer != None:
            print(layer)
            newlay = copy.deepcopy(layer)
            if count <= self.count:
                self.encoder.add(newlay)
                self.encoder.outlayer.update(newlay.getWeights())
                if count == a.count:
                    self.encoder.outlayer.next = None
                    self.decoder.add(Input(self.encoder.outlayer.output))

            else:
                self.decoder.add(newlay)
                self.decoder.outlayer.update(newlay.getWeights())
            layer = layer.getNext()
            count += 1
            
    def train(self,learning_rate=0.0001,epochs=100,loss="mse"):
        
        self.full_model.train(self.X,self.X,lr=learning_rate,epochs=epochs,loss=loss)
        self.finalize_encoder_decoder()
    
    def predict(self,X):
        
        encoded = self.encoder.predict(X)
        decoded = self.decoder.predict(encoded)
        return encoded,decoded, self.full_model.predict(X)


