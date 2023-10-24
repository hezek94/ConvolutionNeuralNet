
import numpy as np 

def meanSqureError(Y_real, Y_train):
    return np.mean(np.power(Y_real - Y_train, 2))
    

def PrimeMSE(Y_real, Y_train):
    return 2*(Y_train - Y_real)/np.size(Y_real)

def BinaryClassifiction_entropy(Y_real, Y_train):
    return np.mean(-Y_real*np.log(Y_train)-(1-Y_real)*np.log(1-Y_train))

def PrimeBinaryClass(Y_real, Y_train):
    return ((1-Y_real)/(1-Y_train)- Y_real/Y_train)/np.size(Y_real)
    