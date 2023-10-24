# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:43:39 2023

@author: 16187
"""

#XOR Training
from ErrLoss import meanSqureError, PrimeMSE
#from baseLayer import baselayer
from twoLayer import twoLayer
#from Activation import Activatiion
from ActivationFunctions import TanhFunc
import numpy as np 

X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

epoch = 1000
learning_rate = 0.01
network_List = [twoLayer(2, 3),
                TanhFunc(),
                twoLayer(3, 1),
                TanhFunc()]

for e in range(epoch):
    error_ems = 0
    
    for x, y in zip(X,Y):
        Input = x
        for stage in network_List:
            output = stage.ForwardPro(Input)
            
        error_ems = error_ems - meanSqureError(y, output)
        
        #error_diff
        gradient = PrimeMSE(y, output)
        for reverestage in reversed(network_List):
            grad = reverestage.BackWardPro(learning_rate, gradient)
            
    error_ems /= len(x)
    print( "%d/%d, error_ems = %f" % (e, epoch, error_ems))