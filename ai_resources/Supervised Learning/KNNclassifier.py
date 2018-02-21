import numpy as np
import sys
sys.path.append('../tools')
import tools

class KNN(): 
    """
        K nearest neighbor classifier. Assign a given vector to a class based on l2 distance
    """
    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        
    def predict(self,k,pred):
        
        classes = []
        for i in range(pred.shape[0]):
            
            predic = pred[i,:]
            print(predic)
            dist = l2distance(self.X,predic)
            indices = dist.argsort()[:k]
            classcounts = y[indices]
            print(classcounts)
            vals,counts = np.unique(classcounts,return_counts=True)
            ind=np.argmax(counts)
            classpick = vals[ind]
            classes.append(classpick)
            
        return np.array(classes)    