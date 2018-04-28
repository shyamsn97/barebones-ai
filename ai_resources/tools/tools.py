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

    return dist

def standardize(X):
	'''
	    z-score standardization, (x -mu)/std(x), standardizing the columns
	'''
	mu = np.mean(X,axis=0)
	mumat = np.outer(mu,np.ones(X.shape[0])).T
	newx = X - mumat

	var = np.sqrt(1/np.var(X,axis=0).reshape((X.shape[0],1)))
	ones = np.ones(X.shape[0]).reshape((1,X.shape[0]))
	varmatrix = var.dot(ones)/X.shape[0]

	return newx.dot(varmatrix)

def cross_val_split_set(X,portion,y=None):
    '''
    use:
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = split_set(X,0.1,y)
    '''
    X = np.array(X)
    y = np.array(y)
    size = int(X.shape[0]*portion)
    indexlist = np.arange(X.shape[0])
    testinds = np.random.choice(indexlist, size, replace=False)
    traininds = np.array([x for x in range(X.shape[0]) if x not in testinds])  
    if np.all(y == None):
        return X[traininds],X[testinds]
    else:
        return X[traininds],X[testinds],y[traininds],y[testinds]


def calc_accuracy(predictions,ytest):
    """
    Calculates accuracy for classification tasks
    """
    acc = ytest - predictions
    return np.where(acc == 0)[0].shape[0]/ytest.shape[0]

def bucket(data):
    """
    buckets continuous data by percentiles
    """
    upper = np.percentile(data,75)
    mid = np.percentile(data,50)
    lower = np.percentile(data,25)
    data[(data <= lower)] = 0
    data[(data > lower) & (data <= mid)] = 1
    data[(data < upper) & (data > mid)] = 2
    data[(data >= upper)] = 3
    return data

def cosine_distance(x,y):
    
    num = np.dot(x,y)
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    return 1 - num/denom

def gen_distance_similarities(X,y,dist='l2'):
    '''
       calculates distance similarities measurements, such as l2, l1, and cosine
    '''
    if dist == 'l2':
        ones = np.ones(y.shape[0]).reshape(y.shape[0],1)
        X = ones.dot(X)
        dist = (y - X)**2
        dist = 1/(1 + np.sqrt(np.sum(dist,axis=1)))
    elif dist == 'l1':
        ones = np.ones(y.shape[0]).reshape(y.shape[0],1)
        X = ones.dot(X)
        dist = np.abs(y - X)
        dist = 1/(np.sum(dist,axis=1) + 1)
    elif dist == 'cosine':
        dist = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            dist[i] = cosine_distance(X,y[i])
    return dist

def generate_similarity_matrix(X,sim_type='l2'):
    
    n = X.shape[0]
    p = X.shape[1]
    
    newmatrix = np.zeros((n,n))
    for i in range(n):
        newmatrix[i] = (gen_distance_similarities(X[i].reshape(1,p),X,sim_type)).reshape(1,n)
    return newmatrix
