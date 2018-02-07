import numpy as np

def compute_covariance(X, correlation=False):
    me = np.mean(X,axis=0)
    meanmat = np.outer(me,np.ones(X.shape[0])).T
    X_cent = X - meanmat
    X_cent = X_cent.dot(X_cent.T)/(X_cent.shape[0] - 1)

    if correlation == True:

        stdev = np.diag(1/np.sqrt(np.diag(X_cent)))
        X_cent = stdev.dot(X_cent).dot(stdev)

    return X_cent


def l2distance(X,y):
	#Gets euclidian distance between every vector in a data matrix X and vector y
    #we can reduce euclidian distance to x^2 + y^2 - 2xy
    X_squared = np.diag(X.dot(X.T)).reshape((X.shape[0],1))
    y_squared =  y.dot(np.outer(y,np.ones(X_squared.shape[0]))).reshape((X_squared.shape[0],1))
    Xy = 2*X.dot(y).reshape(X.shape[0],1)
    
    return np.sqrt(X_squared + y_squared - Xy).reshape((X.shape[0],))
