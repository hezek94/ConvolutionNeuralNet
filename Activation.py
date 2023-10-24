

from baseLayer import baselayer
import numpy as np

class Activatiion(baselayer):
    def __init__(self, activefun, dactivefun):
        self.activefun = activefun
        self.dactivefun = dactivefun
        
    def ForwardPro(self, input):
        self.input = input
        return self.activefun(self.input)
    
    def BackWardPro(self, gradient, learningRate):
        return np.multiply(gradient, self.dactivefun(self.input))

