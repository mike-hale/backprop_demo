from fully_connected import fully_connected
from logistic_regression import logistic_regression
from relu import relu
from convolution import convolution
from softmax import softmax
from maxpool import maxpool
import numpy as np 
from mnist import MNIST
from time import time
from datetime import timedelta


a = convolution((28,28,),1,20,5,1,weights='rand')
a_out = logistic_regression((24,24,20))
a_pool = maxpool((24,24,),20,2,2)
c = fully_connected((12,12,20,),10,weights='rand')
c_out = softmax(10)

def forward(inputs, batched=False):
    out = a.compute(inputs, batched)
    out = a_out.compute(out, batched)
    out = a_pool.compute(out)
    out = c.compute(out, batched)
    out = c_out.compute(out)
    return out

def back_prop(output_error, rate):
    out = c_out.back_prop(output_error)
    out = c.update(out, rate)
    out = a_pool.back_prop(out)
    out = a_out.back_prop(out)
    out = a.update(out, rate)
    return out

def save_weights(epoch):
    if epoch is None:
        a.save_weights('weights/input_conv')
        b.save_weights('weights/second_conv')
        c.save_weights('weights/output_fc')
    else:
        a.save_weights('weights/input_conv_%d' % epoch)
        b.save_weights('weights/second_conv_%d' % epoch)
        c.save_weights('weights/output_fc_%d' % epoch)

def error_grad(output, expected):
    return -1*expected / output

def total_error(output, expected):
    return -1*sum(expected*np.log(output))

def test():
    test_input = np.random.rand(12,12,8)
    test_output = np.random.rand(10)
    while True:
        out = b.compute(test_input)
        out = b_out.compute(out).reshape(4,4,2)
        #out = b_pool.compute(out)
        out = c.compute(out)
        out = c_out.compute(out)
        e_g = error_grad(out,test_output)
        out = c_out.back_prop(e_g)
        out = c.update(out,0.05)
        #out = b_pool.back_prop(out).flatten()
        out = b_out.back_prop(out)
        out = b.update(out,0.05)
        print('Error: ', np.linalg.norm(e_g))

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
            i_data = np.zeros((28,28))
            t_g = 0
            time_start = time()
            for i in range(100):
                i_data = np.array(images[i]).reshape(28,28)
                out = forward(i_data)
                e_g = error_grad(out,lab_data[i])
                back_prop(e_g, 0.1)
                t_g += total_error(out,lab_data[i])
            time_end = time()
            hms_time = str(timedelta(seconds=time_end - time_start))
            print('Epoch ', epoch, ' (', hms_time, ') error: ', t_g)
            epoch += 1
    # Exception handler performs inferences on test data and returns average error
    except KeyboardInterrupt:
        return
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