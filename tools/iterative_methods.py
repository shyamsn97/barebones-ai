import numpy as np

def Batch_Gradient_Descent(X,y,parameters,gradient_func,predict_func,learning_rate=0.001,epochs=200,batch_size=32):
    
    """
    Batch gradient descent
    Parameters:
    	X = np.array() data matrix
    	y = np.array() labels
    	parameters = np.array() weights
    	gradients_func = function to calculate the gradient
		predict_func = function to predict labels with data matrix
    """
    for i in range(epochs):
        
        h = predict_func(X,parameters)
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        sample = 0

        while(sample < y.shape[0]):

            batch_X = X[sample:(sample+batch_size)]
            batch_y = y[sample:(sample+batch_size)]
            batch_h = h[sample:(sample+batch_size)]
            sample += batch_size

            parameters = parameters - learning_rate*gradient_func(parameters,batch_X,batch_y)
            
        print("EPOCHS: " + str(i))
                   
    return parameters

def CnstrPD(n, a):
    RM = np.random.randn(n,n) 
    q,r = np.linalg.qr(RM)
    z = (np.random.rand(n)+a)
    A = q.dot(np.diag(z)).dot(q.T)
    return A


def Jacobi(A,b,e):
    D = np.diag(A) #diagonal entries
    x0 = np.zeros(A.shape[0])
    check = x0
    newcheck = x0 + 2
    iterations = 0
    while((abs(min(newcheck - check)) > e)==True):
        check = newcheck
        r = b - A.dot(check)
        C = r/D
        x = check + C 
        newcheck = x
        iterations += 1
    return x,iterations

def SteepGD(A, b, e):
    x0 = np.zeros(A.shape[0])
    n = len(x0)
    x = x0 #initial vallue
    iterations = 0
    checkx = x + 2
    while((abs(min(checkx - x)) > e)==True):
        checkx = x
        r = b - np.dot(A,x) #compute residual
        t = np.inner(r,r)/np.inner(np.dot(A,r),r) #update step size
        x = x + t*r #update rule
        iterations += 1
    return x, iterations


def CGD(A, b, e):
    x0 = np.zeros(A.shape[0])
    n = len(x0)
    x = x0 #initial vallue
    checkx = x + 2
    r = b - np.dot(A,x)  #initial residual
    p = np.copy(r) #first direction
    iterations = 0
    while((abs(min(checkx - x)) > e)==True):
        checkx = x 
        z = np.dot(A,p)  #compute Ap
        t = np.inner(p,r)/np.inner(p,z)  #compute stepsize
        x = x + t*p #update along direction p
        r = r - t*z #update residual
        b = -np.inner(r,z)/np.inner(p,z)
        p = r + b*p  #pick the next direction
        iterations += 1
    return x, iterations  #kth row stores kth x
