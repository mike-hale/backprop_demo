import numpy as np 
from fully_connected import fully_connected

class convolution:
    def __init__(self, n_inputs, n_outputs, kernel_size, stride):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        output_mul = (n_inputs - kernel_size + 1) // stride
        assert (n_inputs - kernel_size + 1) % stride == 0
        self.fc_layers = []
        for i in range(output_mul):
            self.fc_layers.append(fully_connected(kernel_size, n_outputs))
    
    def compute(self, input_val):
        output_mul = (self.n_inputs - self.kernel_size + 1) // self.stride
        output = np.zeros((output_mul,self.n_outputs))
        for i in range(output_mul):
            output[i] = self.fc_layers[i].compute(input_val[i*self.stride:i*self.stride + self.kernel_size])
        return output

    def back_prop(self, output_error):
        back = np.zeros(self.n_inputs)
        for idx,fc in enumerate(self.fc_layers):
            back[idx*self.stride:idx*self.stride + self.kernel_size] += fc.back_prop(output_error[idx])
        return back
