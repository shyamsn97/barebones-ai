import numpy as np
from tqdm import tqdm

def SGD(dnn,X,y,learning_rate=0.0001,epochs=100,batch_size=1):
    """
    Stochastic Gradient Descent for Neural Networks
    """
    bar = tqdm(np.arange(epochs))
    for i in bar:
        
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        sample = 0
        MSE = 0
        count = 0
        
        while(sample < indices.shape[0]):
            
            batch_X = X[indices[sample:(sample+batch_size)]]
            batch_y = y[indices[sample:(sample+batch_size)]]
            batch_h = dnn.foward(batch_X)
            sample += batch_size
            gradients = dnn.backward_pass(batch_h,batch_y)
            layer = dnn.head.getNext()
            MSE += (1/(batch_size))*np.sum((batch_y - batch_h)**2)
            count += 1
            j = 0

            while np.all(layer != None):

                new_weights = layer.getWeights()
                new_weights[0] -= learning_rate*gradients[j][0]
                new_weights[1] -= learning_rate*gradients[j][1]
                layer.update(new_weights)
                layer = layer.getNext()
                j += 1
                
        string = str(MSE/count)
        bar.set_description("MSE %s" % string)
                   
    