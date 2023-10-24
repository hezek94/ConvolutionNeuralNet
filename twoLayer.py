

from baseLayer import baselayer
import numpy as np


class twoLayer(baselayer):
    def __init__(self, entrance_size, exit_size):
        self.weight = np.random.randn(exit_size, entrance_size)
        self.bias = np.random.randn(exit_size, 1)
        
    def ForwardPro(self, input):
        self.input = input
        return np.dot(self.weight, self.input) + self.bias
    
    
    def BackWardPro(self, gradient, learningRate):
        weight_grad = np.dot(gradient, self.input.T)
        input_out = np.dot(self.weight.T, gradient)
        self.weight =self.weight - learningRate*weight_grad
        self.bias = self.bias - learningRate*gradient
        return input_out