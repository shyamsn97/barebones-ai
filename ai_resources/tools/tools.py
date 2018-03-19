import numpy as np


def sigmoid(x,derivative=False):
	'''
	sigmoid function, set derivative = true to get the derivative
	'''
	if derivative==True:
		return 1/(1+np.e**-(x*1.0))*(1-(1/(1+np.e**-(x*1.0))))
	else:
		return 1/(1+np.e**-(x*1.0))

def compute_covariance(X,col=True, correlation=False):
    '''
        computes covariance using definition 1/n(XXT) - mumuT
        change col to True for 1/n(XTX) - mumuT
        change correlation to True for the correlation matrix
    '''
    if col == False:
        newx = ((X-X.mean(axis=1)).dot(X-X.mean(axis=1)).T)/X.shape[1]
    else:
        newx = ((X-X.mean(axis=0)).T.dot(X-X.mean(axis=0)))/X.shape[0]
    if correlation == True:
        stdev = np.diag(1/np.sqrt(np.diag(newx)))
        newx = stdev.dot(newx).dot(stdev)
    return newx
    
def l2distance(X,y):
    '''
	   gets euclidian distance between vector X and matrix(or vector) y
    '''
    ones = np.ones(y.shape[0]).reshape(y.shape[0],1)
    X = ones.dot(X)
    dist = (y - X)**2
    dist = np.sqrt(np.sum(dist,axis=1))

    # X_squared = np.diag(X.dot(X.T)).reshape((X.shape[0],1))
    # y_squared =  y.dot(np.outer(y,np.ones(X_squared.shape[0]))).reshape((X_squared.shape[0],1))
    # Xy = 2*X.dot(y).reshape(X.shape[0],1)
    # return np.sqrt(X_squared + y_squared - Xy).reshape((X.shape[0],))
    return dist
def standardize(X):
	'''
	    z-score standardization, (x -mu)/std(x), standardizing the rows
	'''
	mu = np.mean(X,axis=0)
	mumat = np.outer(mu,np.ones(X.shape[0])).T
	newx = X - mumat

	var = np.sqrt(1/np.var(X,axis=0).reshape((X.shape[0],1)))
	ones = np.ones(X.shape[0]).reshape((1,X.shape[0]))
	varmatrix = var.dot(ones)/X.shape[0]

	return newx.dot(varmatrix)

