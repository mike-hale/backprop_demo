import numpy as np 

class relu:
    def __init__(self):
        pass

    def compute(self, input_val):
        out = input_val.copy()
        for idx,val in enumerate(out):
            if val < 0:
                out[idx] = 0
        self.last_val = out
        return out

    def back_prop(self, output_error):
        out = self.last_val
        for idx,val in enumerate(out):
            if val > 0:
                out[idx] = 1
        return np.multiply(out,output_error)
