import numpy as np
import sys
sys.path.append('../tools/')
import tools

class DeepNeuralNetwork():
    """
    Deep Neural Net built from scratch
    capable of regression tasks
    TODO: Vanishing/Exploding gradient is an issue right now, very sensitive to initial weight placement.
    """
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.output = 0
        self.layerdims = []
        self.layers = []
        self.weights = []
        self.derivlayers = []
        self.gradients = []
        self.initialized = False
        
    def add(self,nodes):
        if len(self.layerdims) == 0:
        
            self.layerdims.append((self.X.shape[1],nodes))
        else:
            last = self.layerdims[-1]
            self.layerdims.append((last[1],nodes))
    
    def initialize(self,low = -10,high=10,numoutputs=1):
        '''
        make sure to do this after adding all the layers
        '''
        #takes two elements, the first entry is lower bound & second is upper. Default is -10 to 10. 
        #Default number of outputs is 1 output
        self.weights = []
        for i in range(len(self.layerdims)):
            layerdim = self.layerdims[i]
            weight = np.random.uniform(low=low,high=high,size=(layerdim[0],layerdim[1]))
            self.weights.append(weight)
            if (i == (len(self.layerdims)-1)) and self.initialized == False:
                self.weights.append( np.random.uniform(low=low,high=high,size=(layerdim[1],numoutputs)))
                self.layerdims.append((layerdim[1],numoutputs))
        self.initialized = True
        
    def foward(self,X):
        '''
        foward pass
        '''
        self.layers = []
        self.derivlayers = []
        weights = self.weights
        self.layers.append(X)
        i = 0
        val = 0
        while i < len(weights):
            if i == 0:
                val = tools.sigmoid(X.dot(weights[i]))
                derival = tools.sigmoid(X.dot(weights[i]),True)
                self.layers.append(val)
                self.derivlayers.append(derival)
            elif i == (len(weights)-1):
                val = val.dot(weights[i])
                derival = 1
                self.derivlayers.append(derival)
            else:
                derival = tools.sigmoid(val.dot(weights[i]),True)
                val = tools.sigmoid(val.dot(weights[i]))
                self.layers.append(val)
                self.derivlayers.append(derival)
            i += 1
        return val
    
    def calculate_gradients(self):
        self.gradients = []
        y = self.y
        output = self.output
        loss = output - y
        layers = self.layers
        derivlayers = self.derivlayers
        i = len(self.weights)-1

        while i >= 0:
            if(i ==  (len(self.weights)-1)):
                chain = loss
                self.gradients.append(self.layers[i].T.dot(loss))
            else:
                chain = (chain.dot(self.weights[i+1].T))*derivlayers[i]
                self.gradients.append(layers[i].T.dot(chain))
            i-=1
 
        self.gradients = self.gradients[::-1]
            
    def train(self,epochs=1000,learning_rate=0.001):
        '''
        Trains the neural network
        epochs- number of times the network uses the full data set to train
        learning_rate- how much influence the gradient has on the weights, for grad descent
        '''
        if self.initialized == False:
            self.initialize()
        self.output = self.foward(self.X)
        for i in range(epochs):
            MSE = np.sum((self.y - self.output)**2)
            print("MSE at epoch " + str(i) + ": " + str(MSE))
            self.calculate_gradients()
            for i in range(len(self.weights)):
                #print("START")
                #print(self.weights[i])
                self.weights[i] = self.weights[i] - learning_rate*self.gradients[i]
                #print("AFTER")
                #print(self.weights[i])
            self.output = self.foward(self.X)
    
    def predict(self,Xtest):
        weights = self.weights
        i = 0
        val = 0
        while i < len(weights):
            if i == 0:
                val = tools.sigmoid(Xtest.dot(weights[i]))
            elif i == (len(weights)-1):
                val = val.dot(weights[i])
            else:
                val = tools.sigmoid(val.dot(weights[i]))
            i += 1
        return val
    