import numpy as np
import timeit
blog = np.loadtxt('blog.csv', delimiter=',')


#Get the biggest and smallest blog post ID
N_max = np.max(blog)
N_min = np.min(blog)
print N_max, N_min
n = N_max - N_min + 1 #this is the size of the binary matrix
n = int(n)

#binary matrix
BM = np.zeros([n,n])

for i in range(blog.shape[0]): #going through each row of the original data
    blogID = blog[i,0] 
    OutB = blog[i,1]
    BM[int(OutB - 1),int(blogID -1)] = 1.0 #-1 is because the blog ID starts at 1, but python starts at 0
#checks for any websites with no links

def checkforzeroes(B):
    l = []
    for i in range(B.shape[0]):
        vec = B[:,i]
        if sum(vec) == 0:
            l.append(i)
            B[i,i] = 1.0
    print len(l)
    return B
BM = checkforzeroes(BM)

def ColN(A): #normalize each column such that each column sums to 1
    n = A.shape[0]
    Colsum = np.sum(A,axis=0)
    return A/np.outer(np.ones(n),Colsum)
BMP = ColN(BM)

def add_delta(A,delta):
    n = A.shape[0]
    A = delta*A + (1-delta)/n*np.outer(np.ones(n),np.ones(n))
    return A

M = add_delta(BMP,0.85)

def powermethod(A,x0,e): 
    x = x0
    check = x0
    newcheck = x0 + 20
    while((abs(min(newcheck - check)) > e)==True):
        check = newcheck
        x = np.dot(A,x)
        newcheck = x
    return x

newx = (np.zeros(M.shape[0]))
newx[0] = 1.0 #start with first blog
result = powermethod(M,newx,10**-10)
print max(result)


#top 10 delta = 0.85
print (-result).argsort()[:10]

#top 10 delta = 0.75
newM = add_delta(BMP,0.75)
newresult = powermethod(newM,newx,10**-10)
print (-newresult).argsort()[:10]