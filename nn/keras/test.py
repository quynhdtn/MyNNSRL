__author__ = 'quynhdo'
import numpy as np

x = [[[[1,3]],[[3,4]],[[5,6]]]]

x= np.asarray(x)
x=x.reshape(x.shape[0],x.shape[1],x.shape[3])
print(x)
print(x.shape)