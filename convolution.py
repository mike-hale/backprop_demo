import numpy as np 
from fully_connected import fully_connected

class convolution:
    def __init__(self, input_shape, n_outputs, kernel_size, stride, weights=None):
        self.input_shape = input_shape
        self.input_dims = len(input_shape)
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc_shape = []
        self.fc_count = 1
        self.fc_layers = []
        for dim_size in input_shape:
            self.fc_shape.append(1 + (dim_size - kernel_size) // stride)
            self.fc_count *= self.fc_shape[-1]
            assert (dim_size - kernel_size) % stride == 0

        for _ in range(self.fc_count):
            fc_in_shape = ()
            for _ in range(self.input_dims):
                fc_in_shape += (kernel_size,)
            self.fc_layers.append(fully_connected(fc_in_shape, n_outputs, weights=weights))
    
    def compute(self, input_val):
        output = np.zeros((self.fc_count, self.n_outputs))
        for i in range(self.fc_count):
            fc_input = input_val
            i_next = i
            for dim_size in self.fc_shape[::-1]:
                offset = (i_next % dim_size) * self.stride
                i_next = i_next // dim_size
                fc_input = fc_input[offset:offset + self.kernel_size]
            output[i] = self.fc_layers[i].compute(fc_input)
        return output.reshape(tuple(self.fc_shape) + (self.n_outputs,))

    def update(self, output_error, rate):
        back = np.zeros(self.input_shape)
        for idx,fc in enumerate(self.fc_layers):
            i_next = idx
            err_idx = ()
            for dim_size in self.fc_shape[::-1]:
                err_idx += (i_next % dim_size,)
                i_next = i_next // dim_size
            back_update = fc.update(output_error[err_idx], rate)
            if self.input_dims == 3:
                i_offset = (idx*self.stride // self.input_shape[2] // self.input_shape[1]) % self.input_shape[0]
                j_offset = (idx*self.stride // self.input_shape[2]) % self.input_shape[1]
                k_offset = idx*self.stride % self.input_shape[2]
                back[i_offset:i_offset + self.kernel_size,
                     j_offset:j_offset + self.kernel_size,
                     k_offset:k_offset + self.kernel_size] += back_update
            elif self.input_dims == 2:
                i_offset = (idx*self.stride // self.input_shape[1]) % self.input_shape[0]
                j_offset = idx*self.stride % self.input_shape[1]
                back[i_offset:i_offset + self.kernel_size,
                     j_offset:j_offset + self.kernel_size] += back_update
            else:
                offset = idx*self.stride
                print(offset)
                back[offset:offset + self.kernel_size] += back_update

        return back
