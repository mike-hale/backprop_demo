import numpy as np 

class logistic_regression:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def compute(self, input_val, batched=False):
        out = np.zeros(input_val.size)
        for idx,val in enumerate(input_val.flatten()):
            out[idx] = 1/(1 + np.exp(-val))
        if batched:
            self.last_val = np.average(out.reshape(input_val.shape), axis=0)
            return out.reshape(input_val.shape)
        else:
            self.last_val = out.reshape(self.input_shape)
            return self.last_val

    def back_prop(self, output_error):
        if isinstance(self.input_shape, int):
            input_size = self.input_shape
        else:
            input_size = 1
            for dim in self.input_shape:
                input_size *= dim
        out = np.zeros(input_size)
        for idx,val in enumerate(self.last_val.flatten()):
            out[idx] = val*(1 - val)
        return np.multiply(out,output_error.flatten()).reshape(self.input_shape)