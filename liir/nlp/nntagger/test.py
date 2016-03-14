from numpy.random import random

__author__ = 'quynhdo'

import numpy as np
A = [[random() for e in range(4)] for e in range(5)]
A = np.asarray(A)
print(A)
print(A.shape)
A = A[:,::-1]
print(A)