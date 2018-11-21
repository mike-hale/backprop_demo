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


a = convolution((28,28,),1,8,5,1,weights='rand')
a_pool = maxpool((24,24,),8,2,2)
a_out = relu((12,12,8))
b = convolution((12,12,),8,16,5,1,weights='rand')
b_out = relu((8,8,16))
c = fully_connected((8,8,16,),10,weights='rand')
c_out = softmax((10))

def forward(inputs, batched=False):
    out = a.compute(inputs, batched)
    out = a_pool.compute(out)
    out = a_out.compute(out, batched)
    out = b.compute(out)
    out = b_out.compute(out)
    #out = b_pool.compute(out)
    out = c.compute(out, batched)
    out = c_out.compute(out)
    return out

def back_prop(output_error, rate):
    out = c_out.back_prop(output_error)
    out = c.update(out, rate)
    #out = b_pool.back_prop(out)
    out = b_out.back_prop(out)
    out = b.update(out,rate)
    out = a_out.back_prop(out)
    out = a_pool.back_prop(out)
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
    print(len(labels_t))
    try:
        i = 0
        t_g = 0
        correct = 0
        time_start = time()
        while True:
            i_data = np.array(images[i]).reshape(28,28) / 127.5 - 1
            out = forward(i_data)
            if np.argmax(out) == labels[i]:
                correct += 1
            e_g = error_grad(out,lab_data[i])
            back_prop(e_g, 0.01)
            t_g += total_error(out,lab_data[i])
            if i % 100 == 99:
                time_end = time()
                hms_time = str(timedelta(seconds=time_end - time_start))
                print('Epoch ', epoch, ' (', hms_time, ') error: ', t_g, ' (', correct, '/100)')
                if correct == 100:
                    test_error = 0
                    test_correct = 0
                    for j in range(len(images_t)):
                        i_data = np.array(images_t[j]).reshape(28,28) / 127.5 - 1
                        out = forward(i_data)
                        if np.argmax(out) == labels_t[j]:
                            test_correct += 1
                        test_error += total_error(out,lab_data_t[j])
                    print('Test error: ', test_error, '(#correct=', test_correct, ')')
                    return
                correct = 0
                time_start = time_end
                t_g = 0
                epoch += 1

            i = (i + 1) % len(images)
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