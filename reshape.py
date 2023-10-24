import numpy as np
from baseLayer import baselayer

class ReshapeInput(baselayer):
    def __init__(self, shape_input, shape_output):
        self.shape_input = shape_input
        self.shape_output = shape_output
        
    def ForwardPro(self, input):
        return np.reshape(input, self.shape_output)
    
    def BackWardPro(self, gradient,learningRate):
        return np.reshape(gradient, self.shape_input)