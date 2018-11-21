import numpy as np 
from fully_connected import fully_connected
from time import time

class convolution:
    def __init__(self, input_shape, input_depth, n_outputs, kernel_size, stride, weights=None):
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.input_dims = len(input_shape)
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        assert (input_shape[0] - kernel_size) % stride == 0
        assert (input_shape[1] - kernel_size) % stride == 0
        self.output_shape = ((input_shape[0] - kernel_size) // stride + 1, 
                             (input_shape[0] - kernel_size) // stride + 1,)
        if weights == 'rand':
            self.weights = (2/(self.kernel_size**2))*np.random.random_sample((kernel_size, kernel_size, n_outputs, input_depth)) - 1/(self.kernel_size**2)
            self.bias = (2/(self.kernel_size**2))*np.random.random_sample((n_outputs)) - 1/(self.kernel_size**2)
    
    def compute(self, input_val, batched=False):
        self.last_input = input_val
        output = np.zeros(self.output_shape + (self.n_outputs,))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                temp_input = input_val[self.stride*i:self.stride*i + self.kernel_size,
                                       self.stride*j:self.stride*j + self.kernel_size]
                for k in range(self.n_outputs):
                    output[i,j,k] = self.weights[:,:,k,:].flatten().dot(temp_input.flatten()) + self.bias[k]
                        #out = output[:,i,j]
        return output

    def update(self, output_error, rate):
        input_error = np.zeros(self.input_shape + (self.input_depth,))
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.n_outputs):
                    input_error[i*self.stride:i*self.stride + self.kernel_size,
                                j*self.stride:j*self.stride + self.kernel_size] += output_error[i,j,k]*self.weights[:,:,k]
                    last_input = self.last_input[i*self.stride:i*self.stride + self.kernel_size,
                                                 j*self.stride:j*self.stride + self.kernel_size]
                    jacob = output_error[i,j,k]*last_input
                    if self.input_depth == 1:
                        self.weights[:,:,k,0] -= rate*jacob
                    else:
                        self.weights[:,:,k,:] -= rate*jacob
        self.bias -= rate*output_error.sum(axis=(0,1))
        return input_error

    def save_weights(self, filename):
        np.save(filename, self.weights)
