
from baseLayer import baselayer
from Activation import Activatiion
import numpy as np

class TanhFunc(Activatiion): 
    def __init__(self):
        tanActive = lambda x : np.tanh(x)
        DtanActive = lambda x : 1 - np.tanh(x)**2
        super().__init__(tanActive, DtanActive)
        
        
class Sigmoid(Activatiion):
    def __init__(self):
        # sigmod = lambda x : 1/(1-np.exp(-x))
        # sigmod_prime = lambda sigmod : sigmod * (1-sigmod)
        def sigmod(x):
            return 1 / (1 + np.exp(-x))

        def sigmod_prime(x):
            s = sigmod(x)
            return s * (1 - s)

        super().__init__(sigmod, sigmod_prime)
        
        
class Softmax(baselayer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output