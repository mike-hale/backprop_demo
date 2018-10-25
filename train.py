from fully_connected import fully_connected
from logistic_regression import logistic_regression
import numpy as np 

a = fully_connected(2,2,np.array([[.15,.2],[.25,.3]]))
a_out = logistic_regression(2)
b = fully_connected(2,2,np.array([[.4,.45],[.5,.55]]))
b_out = logistic_regression(2)

def forward(inputs):
    out = a.compute(inputs)
    out = a_out.compute(out)
    out = b.compute(out)
    out = b_out.compute(out)
    return out

def back_prop(output_error):
    out = b_out.back_prop(output_error)
    out = b.update(out, 0.05)
    out = a_out.back_prop(out)
    out = a.update(out, 0.05)
    return out

def compute_error(output):
    error = output - np.array([0.01,0.99])
    return error

def train():
    try: 
        while True:
            output = forward(np.array([0.05,0.1]))
            error = compute_error(output)
            back_prop(error)
            print(error[0]**2 + error[1]**2)
    except KeyboardInterrupt:
        return
