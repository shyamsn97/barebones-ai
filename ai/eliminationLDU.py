import numpy as np
from scipy.linalg import det, norm, inv, solve, lu, cholesky, eig, svd
import numpy.linalg as nl


#returns Permutation,Lower Triangular, Upper Triangular
# def ScaledPivot(A):
#     shape = A.shape
#     if shape[0] != shape[1]:
#         raise ValueError("Matrix is not Square")
#     nrow = shape[0]
#     print nrow
#     ncol = shape[1]
#     maxim = []
#     for i in range(nrow):
        
#     for i in range(nrow):
#         indices.append(0)
#     for j in range(ncol):
#         for i in range(nrow):
#             print "placeholder"
 
                    
    

def SolveLower(L,b):
    #first check if lower trianglular
    if np.allclose(L,np.tril(L)) == True:
        check= True
        try:
            inv(L)
            check = True
        except:
            check = False
            return
    else:
        print "Matrix is not Lower Triangular"
        return
    if check == True:
        diagonal = np.diag(L) #diagonal
        shape = L.shape
        i = 0
        b = b.astype(float) #cast as float
        solution = [] #initialize
        while i < shape[0]: #solves from top to bottom
            j = 0
            summed = 0
            while j < i:
                summed = summed + (L[i,j]*solution[j]) #adds all the coeffs*solved values 
                j += 1
            solution.append((b[i] - summed)/diagonal[i]) #subtracts and divides by coefficient that we are solving        
            i += 1

        solution = np.vstack(solution) #stacks into one array
        return solution




def SolveUpper(U,b):
    #first check if upper trianglular
    if np.allclose(U,np.triu(U)) == True:
        check= True
        try:
            inv(U)
            check = True
        except:
            check = False
            return
    else:
        print "Matrix is not Upper Triangular"
        return
    if check == True:
        diagonal = np.diag(U) #diagonal
        shape = U.shape #shape
        i = shape[0]-1 #rows-1
        b = b.astype(float) #cast as float
        solution = [] #initialize
        while i > -1: #solves from bottom to top
            j = shape[0]-1
            count = 0 
            summed = 0
            while j > i:
                summed = summed + (U[i,j]*solution[count])#adds all the coeffs*solved values 
                j -= 1
                count += 1
            solution.append((b[i] - summed)/diagonal[i]) #subtracts and divides by coefficient that we are solving              
            i -= 1
            
        solution = np.vstack(solution) #stacks into one array
        solution = solution[::-1] #reverses
        return solution 







