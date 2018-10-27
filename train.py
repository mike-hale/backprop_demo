from fully_connected import fully_connected
from logistic_regression import logistic_regression
from convolution import convolution
import numpy as np 

a = convolution((13),2,3,2,weights='rand')
a_out = logistic_regression((5,2))
b = fully_connected(10,10,weights='rand')
b_out = logistic_regression(10)

train_in = [1,2,3,4,5,6,7,8,9,10,11,12,13]
train_out = np.random.rand(10)

def forward(inputs):
    out = a.compute(inputs)
    out = a_out.compute(out).flatten()
    out = b.compute(out)
    out = b_out.compute(out)
    return out

def back_prop(output_error):
    out = b_out.back_prop(output_error)
    out = b.update(out, 0.2).reshape(5,2)
    out = a_out.back_prop(out)
    out = a.update(out, 0.2)
    return out

def compute_error(output):
    error = output - np.array(train_out)
    return error

def run():
    while True:
        try:
            while True:
                out = forward(train_in)
                back_prop(compute_error(out))
                total = 0
                for o in compute_error(out):
                    total += o**2
                print(total, ' ', out, '!=', train_out)
        except KeyboardInterrupt:
            try: 
                while True: pass
            except KeyboardInterrupt:
                pass

def train(train_x, train_y, learn_rate, layers):
    try:
        input = train_x
        # Forward stage
        for lay in layers:
            input = lay.compute(input)
        # Compute error
        error = compute_error(input)
        # Backward stage
        for lay in layers[::-1]:
            error = layers.update(error,learn_rate)
    except:
        return
