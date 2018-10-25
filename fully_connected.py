import numpy as np 

class fully_connected:
    def __init__(self, n_inputs, n_outputs, weights=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        if weights is None:
            self.weights = np.zeros((n_outputs, n_inputs))
        else:
            self.weights = weights

    def compute(self, input_val):
        self.last_input = input_val
        return self.weights.dot(input_val)

    def back_prop(self, output_error):
        jacob = np.repeat(np.expand_dims(self.last_input, 0), self.n_outputs, 0)
        # Iterate through rows in weights
        for idx,row in enumerate(jacob):
            jacob[idx] = row*output_error[idx]
        return jacob

    def update(self, output_error, rate):
        jacob = self.back_prop(output_error)
        self.weights -= rate*jacob
        return jacob
        