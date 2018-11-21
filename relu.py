import numpy as np 

class relu:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def compute(self, input_val, batched=False):
        out = input_val.copy().flatten()
        for idx,val in enumerate(out):
            if val < 0:
                out[idx] = 0
        self.last_val = out.reshape(self.input_shape)
        return self.last_val

    def back_prop(self, output_error):
        out = np.zeros((self.input_shape)).flatten()
        for idx,val in enumerate(self.last_val.flatten()):
            if val > 0:
                out[idx] = 1
            else:
                out[idx] = 0
        return np.multiply(out.reshape(self.input_shape),output_error)
