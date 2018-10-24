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
        return self.weights.dot(input_val)

    def back_prop(self, output_error):
        jacob = self.weights
        # Iterate through rows in weights
        for i in range(self.weights.shape[0]):
            jacob[i] *= output_error[i]
        return jacob.sum(0)