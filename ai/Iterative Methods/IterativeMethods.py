

# # Iterative Methods

# Here we will observe the performance of many iterative solutions to Ax = b

# We will assume that our data is Positive Definite, like a Correlation matrix, so we construct it using The basic definition of Diagonalization


import numpy as np
import timeit



def CnstrPD(n, a):
    RM = np.random.randn(n,n) 
    q,r = np.linalg.qr(RM)
    z = (np.random.rand(n)+a)
    A = q.dot(np.diag(z)).dot(q.T)
    return A



A = CnstrPD(5000, 0.1) #5000x5000 matrix
b = np.random.randn(5000)


# First we'll look at Gaussian Elimination


start = timeit.default_timer()
x_g = np.linalg.solve(A, b)
end = timeit.default_timer()
et = end - start
estimate = A.dot(x_g)
mse = (sum(b - estimate))**2
print "Finished in", et, "seconds"
print "Iterations: ", 1
print "First 10 entries of solution is", x_g[:10]
print "Mean Squared Error: ", mse


# Now we can look at Jacobi iteration


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



start = timeit.default_timer()
x_j, iterations = Jacobi(A,b,10**-8)
end = timeit.default_timer()
et = end - start
estimate = A.dot(x_j)
mse = (sum(b - estimate))**2
print "Finished in", et, "seconds"
print "Iterations: ", iterations
print "First 10 entries of solution is", x_j[:10]
print "Mean Squared Error: ", mse


# Here is the method of Steepest descent

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


start = timeit.default_timer()
x_gd, iterations = SteepGD(A,b,10**-8)
end = timeit.default_timer()
et = end - start
estimate = A.dot(x_gd)
mse = (sum(b - estimate))**2
print "Finished in", et, "seconds"
print "Iterations: ", iterations
print "First 10 entries of solution is", x_gd[:10]
print "Mean Squared Error: ", mse


# Finally, we'll look at how Conjugate Gradient Descent performs


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


start = timeit.default_timer()
x_cgd, iterations = CGD(A,b,10**-8)
end = timeit.default_timer()
et = end - start
estimate = A.dot(x_cgd)
mse = (sum(b - estimate))**2
print "Finished in", et, "seconds"
print "Iterations: ", iterations
print "First 10 entries of solution is", x_cgd[:10]
print "Mean Squared Error: ", mse


# Accuracy measured by Mean Squared Error
# Conjugate Gradient Descent is the quickest algorithm, with the second best accuracy
# Jacobi is the second quickest algorithm with the second worst accuracy
# Steepest Gradient Descent is the 3rd quickest algorithm with the worst accuracy
# Gaussian elimination is the slowest algorithm with the highest algorithm
