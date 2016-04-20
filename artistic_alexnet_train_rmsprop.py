#THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' python artistic_alexnet_train_rmsprop.py

import os
import sys
import timeit

import matplotlib.pyplot as plt

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample, pool
from theano.tensor.nnet import conv2d

from pylearn2.expr.normalize import CrossChannelNormalization

params_path = 'pretrained_weights/parameters_releasing'


class ConvPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape, f_params_w, f_params_b, lrn=False, t_style=None, t_content=None, convstride=1, padsize =0, group=1, poolsize = 3, poolstride = 1):

        self.input = input

        if t_style is not None:
            self.t_style = np.asarray(np.load(t_style),dtype=theano.config.floatX)

        if t_content is not None:
            self.t_content = np.asarray(np.load(t_content),dtype=theano.config.floatX)

        if lrn is True:
            self.lrn_func = CrossChannelNormalization()

        if group == 1:
            self.W = theano.shared(np.asarray(np.transpose(np.load(os.path.join(params_path,f_params_w)),(3,0,1,2)),dtype=theano.config.floatX), borrow=True)
            self.b = theano.shared(np.asarray(np.load(os.path.join(params_path,f_params_b)),dtype=theano.config.floatX), borrow=True)
            conv_out = conv2d(input=self.input,filters=self.W,filter_shape=filter_shape,border_mode = padsize,subsample=(convstride, convstride),filter_flip=True)

        elif group == 2:
            self.filter_shape = np.asarray(filter_shape)
            self.image_shape = np.asarray(image_shape)
            self.filter_shape[0] = self.filter_shape[0] / 2
            self.filter_shape[1] = self.filter_shape[1] / 2
            self.image_shape[1] = self.image_shape[1] / 2
            self.W0 = theano.shared(np.asarray(np.transpose(np.load(os.path.join(params_path,f_params_w[0])),(3,0,1,2)),dtype=theano.config.floatX), borrow=True)
            self.W1 = theano.shared(np.asarray(np.transpose(np.load(os.path.join(params_path,f_params_w[1])),(3,0,1,2)),dtype=theano.config.floatX), borrow=True)
            self.b0 = theano.shared(np.asarray(np.load(os.path.join(params_path,f_params_b[0])),dtype=theano.config.floatX), borrow=True)
            self.b1 = theano.shared(np.asarray(np.load(os.path.join(params_path,f_params_b[1])),dtype=theano.config.floatX), borrow=True)
            conv_out0 = conv2d(input=self.input[:,:self.image_shape[1],:,:],filters=self.W0,filter_shape=tuple(self.filter_shape),border_mode = padsize,subsample=(convstride, convstride),filter_flip=True) + self.b0.dimshuffle('x', 0, 'x', 'x')
            conv_out1 = conv2d(input=self.input[:,self.image_shape[1]:,:,:],filters=self.W1,filter_shape=tuple(self.filter_shape),border_mode = padsize,subsample=(convstride, convstride),filter_flip=True) + self.b1.dimshuffle('x', 0, 'x', 'x')
            conv_out = T.concatenate([conv_out0, conv_out1],axis=1)

        else:
            raise AssertionError()

        relu_out = T.maximum(conv_out, 0)
        if poolsize != 1:
            self.output = pool.pool_2d(input=relu_out,ds=(poolsize,poolsize),ignore_border=True, st=(poolstride,poolstride),mode='average_exc_pad')
        else:
            self.output = relu_out

        if lrn is True:
            self.output = self.lrn_func(self.output)

    def style_error(self):
        gram_matrix_ori = T.dot(self.t_style.reshape((self.t_style.shape[1],self.t_style.shape[2]*self.t_style.shape[3])),self.t_style.reshape((self.t_style.shape[1],self.t_style.shape[2]*self.t_style.shape[3])).T)
        gram_matrix_gen = T.dot(self.output.reshape((self.t_style.shape[1],self.t_style.shape[2]*self.t_style.shape[3])),self.output.reshape((self.t_style.shape[1],self.t_style.shape[2]*self.t_style.shape[3])).T)
        return T.sum(T.sum((gram_matrix_gen-gram_matrix_ori)**2))/(4.0*(self.t_style.shape[1]**2)*((self.t_style.shape[2]*self.t_style.shape[3])**2))

    def content_error(self):
        return T.sum(T.sum(T.sum((self.output-self.t_content)**2)))/2.0

def evaluate_alexnet(batch_size=1):

    rng = np.random.RandomState(23455)

    input_img = np.random.normal(0.0, 1.0, size=(1,3,227,227)).astype(np.float32)

    lr = T.fscalar('lr')

    x = theano.shared(input_img,borrow=True)
    print('... building the model')

    layer1_input = x.reshape((batch_size, 3, 227, 227))

    convpool_layer1 = ConvPoolLayer(input=layer1_input, image_shape=(batch_size, 3, 227, 227), filter_shape=(96, 3, 11, 11), f_params_w='W_0_65.npy', f_params_b='b_0_65.npy', t_style = 'cnn_features/van_gogh_starry_night_1.npy', t_content = 'cnn_features/kaist_n1_1.npy', lrn=True, convstride=4, padsize=0, group=1, poolsize=3, poolstride=2)

    convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,image_shape=(batch_size, 96, 27, 27),filter_shape=(256, 96, 5, 5), f_params_w=['W0_1_65.npy','W1_1_65.npy'], t_style = 'cnn_features/van_gogh_starry_night_2.npy', lrn=True, f_params_b=['b0_1_65.npy','b1_1_65.npy'], convstride=1, padsize=2, group=2, poolsize=3, poolstride=2)

    convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,image_shape=(batch_size, 256, 13, 13),filter_shape=(384, 256, 3, 3), f_params_w='W_2_65.npy', f_params_b='b_2_65.npy', t_style = 'cnn_features/van_gogh_starry_night_3.npy',convstride=1, padsize=1, group=1,poolsize=1, poolstride=0)

    convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,image_shape=(batch_size, 384, 13, 13),filter_shape=(384, 384, 3, 3), f_params_w=['W0_3_65.npy','W1_3_65.npy'], t_style = 'cnn_features/van_gogh_starry_night_4.npy', f_params_b=['b0_3_65.npy','b1_3_65.npy'],convstride=1, padsize=1, group=2,poolsize=1, poolstride=0)

    convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,image_shape=(batch_size, 384, 13, 13),filter_shape=(256, 384, 3, 3), f_params_w=['W0_4_65.npy','W1_4_65.npy'], t_style = 'cnn_features/van_gogh_starry_night_5.npy', f_params_b=['b0_4_65.npy','b1_4_65.npy'], convstride=1, padsize=1, group=2,poolsize=3, poolstride=2)

    cost= 0.2*(convpool_layer1.style_error() + convpool_layer2.style_error() +convpool_layer3.style_error() + convpool_layer4.style_error() +convpool_layer5.style_error()) + 0.00002*convpool_layer1.content_error()

    img_out = theano.function([],x)

    print('... train')

    params = x

    grads = T.grad(cost, params)

    #####RMSprop
    decay = 0.9
    max_scaling=1e5
    epsilon = 1. / max_scaling

    vels = theano.shared(params.get_value() * 0.) 

    new_mean_squared_grad = (decay * vels + (1 - decay) * T.sqr(grads))
    rms_grad_t = T.sqrt(new_mean_squared_grad)
    delta_x_t = - lr * grads / rms_grad_t

    updates=[]
    updates.append((params,params + delta_x_t))
    updates.append((vels,new_mean_squared_grad))


    train_model = theano.function([lr],[cost],updates=updates)

    n_epochs = 3000

    img_mean = np.load('pretrained_weights/img_mean.npy')
    img_mean = np.transpose(img_mean,(1,2,0))

    tmp = np.transpose(np.squeeze(input_img),(1,2,0))

    recon = tmp+img_mean[16:16+227,16:16+227,:]

    fig = plt.figure()
    fig_handle = plt.imshow(recon.astype(np.uint8))
    fig.show()

    learning_rate = 1.0
    schedules = [5000,12000]
    lr_phase = 0
    for i in xrange(n_epochs):
        img_gen = np.transpose(np.squeeze(img_out()),(1,2,0))
        recon = img_gen+img_mean[16:16+227,16:16+227,:]
        fig_handle.set_data(recon.astype(np.uint8))
        fig.canvas.draw()
    
        if i==schedules[lr_phase]:
            lr_phase+=1
            learning_rate*=0.1
            results = img_out()
            np.save('results.npy',results)

        print 'lr : ', learning_rate
        print train_model(learning_rate)[0], ' / epochs : ', i

    results = img_out()

    np.save('results.npy',results)


if __name__ == '__main__':
    evaluate_alexnet()


def experiment(state, channel):
    evaluate_alexnet(state.learning_rate, dataset=state.dataset)
