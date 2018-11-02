import numpy as np 

class fully_connected:
    def __init__(self, input_shape, n_outputs, weights=None):
        self.input_shape = input_shape
        n_inputs = 1
        for dim_size in input_shape:
            n_inputs *= dim_size
        self.n_outputs = n_outputs
        if weights is None:
            self.weights = np.zeros((n_outputs, n_inputs))
        elif weights == 'rand':
            self.weights = 2*np.random.random_sample((n_outputs, n_inputs)) - 1
        else:
            self.weights = weights

    def compute(self, input_val, batched=False):
        if batched:
            self.last_input = np.average(input_val, axis=0)
            ret = np.zeros((len(input_val), self.weights.shape[0]))
            for idx,single_input in enumerate(input_val):
                ret[idx] = self.weights.dot(single_input.flatten())
            return ret
        else:
            self.last_input = input_val
            return self.weights.dot(input_val.flatten())

    def back_prop(self, output_error):
        jacob = np.repeat(np.expand_dims(self.last_input.flatten(), 0), self.n_outputs, 0)
        # Iterate through rows in weights
        for idx,row in enumerate(jacob):
            jacob[idx] = row*output_error[idx]
        return jacob

    def update(self, output_error, rate):
        jacob = self.back_prop(output_error)
        self.weights -= rate*jacob
        return jacob.sum(0).reshape(self.input_shape)
        