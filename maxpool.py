import numpy as np 

class maxpool:
    def __init__(self, input_shape, input_depth, kernel, stride):
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.kernel = kernel 
        self.stride = stride
        self.output_shape = ((input_shape[0] - kernel) // stride + 1, (input_shape[1] - kernel) // stride + 1,)
        assert (input_shape[0] - kernel) % stride == 0
        assert (input_shape[1] - kernel) % stride == 0

    def compute(self, input_val):
        self.last_val = input_val
        output = np.zeros(self.output_shape + (self.input_depth,))
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                input_range = input_val[i*self.kernel:i*self.kernel + self.stride,
                                        j*self.kernel:j*self.kernel + self.stride]
                output[i,j] = input_range.max(axis=(0,1))
        return output

    def back_prop(self, output_error):
        input_error = np.zeros(self.input_shape + (self.input_depth,))
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                argmaxs = self.last_val[i*self.kernel:i*self.kernel + self.stride,
                                        j*self.kernel:j*self.kernel + self.stride].reshape(self.kernel*self.kernel,self.input_depth).argmax(axis=0)
                for m in range(self.input_depth):
                    m_x = argmaxs[m] // self.kernel
                    m_y = argmaxs[m] % self.kernel
                    input_error[i*self.kernel + m_x,j*self.kernel + m_y,m] = output_error[i,j,m]
        return input_error
                
                