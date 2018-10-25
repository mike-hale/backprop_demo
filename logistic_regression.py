import numpy as np 

class logistic_regression:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

    def compute(self, input_val):
        out = np.zeros(self.n_inputs)
        for idx,val in enumerate(input_val):
            out[idx] = 1/(1 + np.exp(-val))
        self.last_val = out
        return out

    def back_prop(self, output_error):
        out = np.zeros(self.n_inputs)
        for idx,val in enumerate(self.last_val):
            out[idx] = val*(1 - val)
        return np.multiply(out,output_error)