from numpy.random import random

__author__ = 'quynhdo'

import numpy as np
A = [[random() for e in range(4)] for e in range(5)]
print(A)
A=np.matrix(A)
print(A.shape)
B = A[1,:]
M=[]
B=B.tolist()
M.append(B)
print (np.asarray(M).shape)
M=np.asarray(M).reshape(1,4)
print(M)
