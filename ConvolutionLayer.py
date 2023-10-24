import numpy as np

from scipy import signal
from baseLayer import baselayer


class Convolution(baselayer):
    def __init__(self, input_shape, kernal_size, depth_kernel):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.depthkernel = depth_kernel
        self.output_shape = (depth_kernel, input_height-kernal_size+1, \
            input_width-kernal_size+1)
        self.kernel_shape = (depth_kernel, input_depth, kernal_size, kernal_size)
        #shape = (num_of layer, depth, width and height) *self.. means unpacking
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def ForwardPro(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depthkernel):
            for j in range(self.input_depth):
                self.output += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output
    
    def BackWardPro(self, gradient, learningRate):
        #we instantiate the diff of both input and kernal with error as 0
        kernal_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape) 
        
        for i in range(self.depthkernel):
            for j in range(self.input_depth):
                kernal_gradient[i,j] = signal.correlate2d(self.input[j],gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(gradient[i], self.kernels[i,j], "full")     
        self.kernels -= learningRate*kernal_gradient
        self.biases -= learningRate*gradient
        return input_gradient