import numpy as np
import matplotlib.pyplot as plt

class softmax:
    def __init__(self, inputs):
        self.inputs = inputs
    
    def compute(self):
        exps = np.zeros(self.inputs.shape)
        for idx,item in enumerate(self.inputs):
            exps[idx] = np.exp(item)
        exps = exps / sum(exps)
        return exps

    def back_prop(self, output_error):
        shape = self.inputs.shape + self.inputs.shape
        jacob = np.zeros(shape)
        exps = self.compute()
        for i,e_i in enumerate(exps):
            for j,e_j in enumerate(exps):
                if i == j:
                    jacob[i,j] = e_i*(1 - e_j)
                else:
                    jacob[i,j] = -e_i*e_j
        return np.multiply(output_error, jacob.sum(1))
