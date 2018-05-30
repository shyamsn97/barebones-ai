import numpy as np
from tqdm import tqdm

def SGD(dnn,X,y,learning_rate=0.0001,epochs=100,batch_size=1):
    """
    Stochastic Gradient Descent for Neural Networks
    """
    for i in tqdm(range(epochs)):
        
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
            i = 0
            MSE += np.sum((batch_y - batch_h)**2)
            count += 1
            while np.all(layer != None):
                layer.update(layer.getWeights() - learning_rate*gradients[i])
                layer = layer.getNext()
                i += 1
                
        print("MSE: " + str(MSE/count))
                   
    