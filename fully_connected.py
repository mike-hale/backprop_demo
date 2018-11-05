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

    def update(self, output_error, rate):
        input_error = self.weights.T.dot(output_error)
        jacob = output_error[:,np.newaxis].dot(self.last_input.flatten()[np.newaxis,:])
        self.weights -= rate*jacob
        return input_error.reshape(self.input_shape)

    def save_weights(self, filename):
        np.save(filename, self.weights)
        