from fully_connected import fully_connected
from logistic_regression import logistic_regression
from convolution import convolution
import numpy as np 
from mnist import MNIST

a = convolution((29,29,),1,3,5,2,weights='rand')
a_out = logistic_regression((13,13,3))
b = convolution((13,13,),3,3,4,1,weights='rand')
b_out = logistic_regression((10,10,3))
c = convolution((10,10,),3,3,3,1,weights='rand')
c_out = logistic_regression((8,8,3,))
d = fully_connected((8,8,3,),10,weights='rand')
d_out = logistic_regression((10))

train_in = [1,2,3,4,5,6,7,8,9,10,11,12,13]
train_out = np.random.rand(10)

def forward(inputs, batched=False):
    out = a.compute(inputs, batched)
    out = a_out.compute(out, batched)
    out = b.compute(out, batched)
    out = b_out.compute(out, batched)
    out = c.compute(out, batched)
    out = c_out.compute(out, batched)
    out = d.compute(out, batched)
    out = d_out.compute(out, batched)
    return out

def back_prop(output_error, rate):
    out = d_out.back_prop(output_error)
    out = d.update(out, rate)
    out = c_out.back_prop(out)
    out = c.update(out, rate)
    out = b_out.back_prop(out)
    out = b.update(out, rate)
    out = a_out.back_prop(out)
    out = a.update(out, rate)
    return out

def error_grad(output, expected):
    return output - expected

def total_error(output, expected):
    return np.linalg.norm(output - expected)

def main():
    data = MNIST('../dataset_imgs')
    images, labels = data.load_training()
    images_t, labels_t = data.load_testing()
    lab_data = np.zeros((len(labels),10))
    lab_data_t = np.zeros((len(labels_t),10))
    for idx,l in enumerate(labels):
        lab_data[idx,l] = 1
    for idx,l in enumerate(labels_t):
        lab_data_t[idx,l] = 1
    epoch = 0
    try:
        while True:
            i_data = np.zeros((29,29))
            t_g = 0
            for i in range(len(images)):
                i_data[:28,:28] = np.array(images[i]).reshape(28,28)
                out = forward(i_data)
                e_g = error_grad(out,lab_data[i])
                back_prop(e_g, 0.05)
                t_g += total_error(out,lab_data[i])
            print('Epoch ', epoch, ' error: ', t_g)
            epoch += 1
    except KeyboardInterrupt:
        t_g = 0
        for i in range(len(images_t)):
            i_data[:28,:28] = np.array(images[i]).reshape(28,28)
            out = forward(i_data)
            t_g += total_error(out,lab_data[i])
        print('Average test error: ', t_g / len(images_t))
        

def train(train_x, train_y, learn_rate, layers):
    try:
        input = train_x
        # Forward stage
        for lay in layers:
            input = lay.compute(input)
        # Compute error
        error = error_grad(input)
        # Backward stage
        for lay in layers[::-1]:
            error = layers.update(error,learn_rate)
    except:
        return

if __name__=='__main__':
    main()