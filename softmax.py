import numpy as np
import matplotlib.pyplot as plt

class softmax:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
    
    def compute(self, input_val):
        exps = np.zeros(self.n_inputs)
        for idx,item in enumerate(input_val):
            exps[idx] = np.exp(item)
        exps = exps / sum(exps)
        self.last_val = exps
        return exps

    def back_prop(self, output_error):
        shape = (self.n_inputs, self.n_inputs)
        jacob = np.zeros(shape)
        exps = self.last_val
        for i,e_i in enumerate(exps):
            for j,e_j in enumerate(exps):
                if i == j:
                    jacob[i,j] = e_i*(1 - e_j)
                else:
                    jacob[i,j] = -e_i*e_j
        return jacob.dot(output_error)
