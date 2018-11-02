import numpy as np 
from fully_connected import fully_connected

class convolution:
    def __init__(self, input_shape, input_depth, n_outputs, kernel_size, stride, weights=None):
        self.input_shape = input_shape
        self.input_depth = input_depth
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
            fc_in_shape += (input_depth,)
            self.fc_layers.append(fully_connected(fc_in_shape, n_outputs, weights=weights))
    
    def compute(self, input_val, batched=False):
        if batched:
            input_val = input_val.transpose(tuple(range(1,input_val.ndim)) + (0,))
            output = np.zeros((input_val.shape[-1],self.fc_count, self.n_outputs))
        else:
            output = np.zeros((self.fc_count, self.n_outputs))
        for i in range(self.fc_count):
            if self.input_dims == 1:
                offset = i * self.stride
                fc_input = input_val[offset:offset + self.kernel_size]
            elif self.input_dims == 2:
                i_x = i % self.fc_shape[0]
                i_y = i // self.fc_shape[1]
                offset_x = i_x * self.stride
                offset_y = i_y * self.stride
                fc_input = input_val[offset_x:offset_x + self.kernel_size,
                                     offset_y:offset_y + self.kernel_size]
            elif self.input_dims == 3:
                i_x = i % self.fc_shape[0]
                i_y = (i // self.fc_shape[0]) % self.fc_shape[1]
                i_z = (i // self.fc_shape[0]) // self.fc_shape[1]
                offset_x = i_x * self.stride
                offset_y = i_y * self.stride
                offset_z = i_z * self.stride
                fc_input = input_val[offset_x:offset_x + self.kernel_size,
                                     offset_y:offset_y + self.kernel_size,
                                     offset_z:offset_z + self.kernel_size]
            if batched:
                output[:,i] = self.fc_layers[i].compute(fc_input.transpose((fc_input.ndim - 1,) + tuple(range(fc_input.ndim - 1))), batched)
            else:
                output[i] = self.fc_layers[i].compute(fc_input, batched)
        if batched:
            return output.reshape((input_val.shape[-1],) + tuple(self.fc_shape) + (self.n_outputs,))
        else:
            return output.reshape(tuple(self.fc_shape) + (self.n_outputs,))

    def update(self, output_error, rate):
        back = np.zeros(self.input_shape + (self.input_depth,))
        for idx,fc in enumerate(self.fc_layers):
            i_next = idx
            err_idx = ()
            for dim_size in self.fc_shape[::-1]:
                err_idx += (i_next % dim_size,)
                i_next = i_next // dim_size
            back_update = fc.update(output_error[err_idx], rate)
            if self.input_dims == 3:
                i_x = idx % self.fc_shape[0]
                i_y = (idx // self.fc_shape[0]) % self.fc_shape[1]
                i_z = (idx // self.fc_shape[0]) // self.fc_shape[1]
                offset_x = i_x * self.stride
                offset_y = i_y * self.stride
                offset_z = i_z * self.stride
                back[offset_x:offset_x + self.kernel_size,
                     offset_y:offset_y + self.kernel_size,
                     offset_z:offset_z + self.kernel_size] += back_update
            elif self.input_dims == 2:
                i_x = idx % self.fc_shape[0]
                i_y = idx // self.fc_shape[1]
                offset_x = i_x * self.stride
                offset_y = i_y * self.stride
                back[offset_x:offset_x + self.kernel_size,
                     offset_y:offset_y + self.kernel_size] += back_update
            else:
                offset = idx * self.stride
                back[offset:offset + self.kernel_size] += back_update

        return back
