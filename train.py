from fully_connected import fully_connected
from logistic_regression import logistic_regression
from relu import relu
from convolution import convolution
from softmax import softmax
import numpy as np 
from mnist import MNIST
from time import time
from datetime import timedelta

a = convolution((29,29,),1,3,5,3,weights='rand')
a_out = logistic_regression((9,9,3))
b = convolution((9,9,),3,4,3,1,weights='rand')
b_out = logistic_regression((7,7,4))
c = fully_connected((7,7,4,),10,weights='rand')
c_out = softmax(10)

def forward(inputs, batched=False):
    out = a.compute(inputs, batched)
    out = a_out.compute(out, batched)
    out = b.compute(out, batched)
    out = b_out.compute(out, batched)
    out = c.compute(out, batched)
    out = c_out.compute(out)
    return out

def back_prop(output_error, rate):
    out = c_out.back_prop(output_error)
    out = c.update(out, rate)
    out = b_out.back_prop(out)
    out = b.update(out, rate)
    out = a_out.back_prop(out)
    out = a.update(out, rate)
    return out

def save_weights(epoch):
    if epoch is None:
        a.save_weights('weights/input_conv.np')
        b.save_weights('weights/second_conv.np')
        c.save_weights('weights/output_fc.np')
    else:
        a.save_weights('weights/input_conv_%d.np' % epoch)
        b.save_weights('weights/second_conv_%d.np' % epoch)
        c.save_weights('weights/output_fc_%d.np' % epoch)

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
            time_start = time()
            for i in range(len(images)):
                i_data[:28,:28] = np.array(images[i]).reshape(28,28)
                out = forward(i_data)
                e_g = error_grad(out,lab_data[i])
                back_prop(e_g, 0.05)
                t_g += total_error(out,lab_data[i])
            time_end = time()
            hms_time = str(timedelta(seconds=time_end - time_start))
            print('Epoch ', epoch, ' (', hms_time, ') error: ', t_g)
            save_weights(epoch)
            epoch += 1
    # Exception handler performs inferences on test data and returns average error
    except KeyboardInterrupt:
        t_g = 0
        for i in range(len(images_t)):
            i_data[:28,:28] = np.array(images[i]).reshape(28,28)
            out = forward(i_data)
            t_g += total_error(out,lab_data[i])
        print('Average test error: ', t_g / len(images_t))
        save_weights(None)
        

def train(train_x, train_y, learn_rate, layers):
    try:
        input = train_x
        # Forward stage
        for lay in layers:
            input = lay.compute(input)
        # Compute error
        error = error_grad(input, input)
        # Backward stage
        for lay in layers[::-1]:
            error = layers.update(error,learn_rate)
    except:
        return

if __name__=='__main__':
    main()