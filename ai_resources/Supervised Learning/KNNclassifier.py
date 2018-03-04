import numpy as np
import sys
sys.path.append('../tools')
import tools

class KNN(): 
    """
        K nearest neighbor classifier. Assign a given vector to a class based on l2 distance
        Parameters:
        numpy array X: data matrix
        numpy array y: labels/classes
    """
    def __init__(self,X,y):
        
        self.X = X
        self.y = y
        
    def predict(self,k,pred):
        
        classes = []
        for i in range(pred.shape[0]):
            predic = pred[i,:]
            nvec = self.X[i,:]
            yval = self.y[i]
            self.y = np.delete(self.y,i)
            self.X = np.delete(self.X, (i),axis=0)
            dist = tools.l2distance(self.X,predic)
            self.X = np.insert(self.X, i, nvec, 0) 
            indices = dist.argsort()[:k]
            classcounts = y[indices]
            self.y = np.insert(self.y,i,yval)
            vals,counts = np.unique(classcounts,return_counts=True)
            ind=np.argmax(counts)
            classpick = vals[ind]
            classes.append(classpick)
            
        return np.array(classes)   