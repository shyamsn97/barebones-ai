import numpy as np
import pandas as pd
import tensorflow as tf
from keras import *
from keras.layers import *
from keras import backend as K


class MultiChannelCNN():
    """
    Multi Channel Convolutional Neural network built with keras
    Can customize how many channels, kernel size for each channel, and the number of filters
    """
    saved = 0
    
    def __init__(self, X,y,models=[]):
        self.X = np.expand_dims(X, axis=2) #need to add an extra column, 1d convolution needs to "slide" accross
        self.y = y
        self.models = models
        
    #channels is an integer, number of channels
    #kernel size is a list of dimensions for the kernels
    def createmodel(self,channels,kernel_size,num_filters):
        
        K.clear_session()

        inputlayers = {}
        layers = {}
        flats = {}
        length = self.X.shape[1]
        for i in range(channels):
            print i
            inputlayers["input"+ str(i)] = Input(shape=(length,1))
            print inputlayers["input"+str(i)]
            layers["conv" + str(i)] = Conv1D(filters=num_filters,input_shape=(length, 1), kernel_size=kernel_size[i], activation='relu')(inputlayers["input" + str(i)])
            layers["dropout" + str(i)] =  Dropout(0.5)(layers["conv" + str(i)])
            layers["pool" + str(i)] = MaxPooling1D(pool_size=4)(layers["dropout" + str(i)])
            flats["flat" + str(i)] = Flatten()(layers["pool" + str(i)])
        
        merge = concatenate(list(flats.values()))
        dense = Dense(10, activation='relu')(merge)
        outputs = Dense(10, activation='sigmoid')(dense)
        model = Model(inputs=list(inputlayers.values()), outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.models.append(model)
    
    #train using custom params
    def train(self,model,epochs,channels,batch_size):
        
        inp = []
        for i in range(channels):
            inp.append(self.X)
        
#         model.fit(inp, self.y,validation_split=0.1, epochs=epochs, batch_size=batch_size,verbose=1)
        model.fit(inp, self.y,validation_split=0.1, epochs=epochs,verbose=1)

        
        if MultiChannelCNN.saved < 1:
            model.save('multichannelcnn.h5')
        else:
            print("Already Saved")
        loss, acc = model.evaluate([self.X,self.X,self.X], self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
        return model
    
    #predict
    def predict(self,model,data):
        #model = load_model('multichannelcnn.h5')
        predicts = model.predict(data)

        return predicts 