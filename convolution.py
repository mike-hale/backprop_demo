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
                             (input_shape[0] - kernel_size) // stride + 1)
        if weights == 'rand':
            self.weights = 2*np.random.random_sample((kernel_size, kernel_size, n_outputs, input_depth)) - 1
    
    def compute(self, input_val, batched=False):
        self.last_input = input_val
        output = np.zeros((self.n_outputs,) + self.output_shape)
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                temp_input = input_val[self.stride*i:self.stride*i + self.kernel_size,
                                       self.stride*j:self.stride*j + self.kernel_size]
                for n in range(self.kernel_size):
                    for m in range(self.kernel_size):
                        output[:,i,j] += self.weights[n,m].dot(input_val[self.stride*i + n,self.stride*j + m])[0]
                        #out = output[:,i,j]
        return output

    def update(self, output_error, rate):
        input_error = np.zeros(self.input_shape + (self.input_depth,))
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        input_error[i*self.stride + m,j*self.stride + n] += self.weights[m,n].T.dot(output_error[i,j])
                        if self.input_depth == 1:
                            jacob = self.last_input[i+n,j+m]*output_error[i,j,:,np.newaxis]
                            self.weights[m,n] -= rate*jacob
                        else:
                            jacob = output_error[i,j,:,np.newaxis].dot(self.last_input[np.newaxis,i+n,j+m])
                            self.weights[m,n] -= rate*jacob
        return input_error

    def save_weights(self, filename):
        np.save(filename, self.weights)
