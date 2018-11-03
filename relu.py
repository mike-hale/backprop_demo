import numpy as np 

class relu:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def compute(self, input_val, batched=False):
        out = input_val.copy().flatten()
        for idx,val in enumerate(out):
            if val < 0:
                out[idx] = 0.1*val
        self.last_val = out.reshape(self.input_shape)
        return self.last_val

    def back_prop(self, output_error):
        out = self.last_val.flatten()
        for idx,val in enumerate(out):
            if val > 0:
                out[idx] = 1
        return np.multiply(out.reshape(self.input_shape),output_error)
