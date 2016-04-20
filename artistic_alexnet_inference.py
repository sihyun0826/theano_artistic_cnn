#THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' python artistic_alexnet_inference.py

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from pylearn2.expr.normalize import CrossChannelNormalization

params_path = 'pretrained_weights/parameters_releasing'

class ConvPoolLayer(object):

    def __init__(self, input, filter_shape, image_shape, f_params_w, f_params_b, lrn=False, convstride=1, padsize =0, group=1, poolsize = 3, poolstride = 1):

        assert image_shape[1] == filter_shape[1]
        self.input = input

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

def evaluate_alexnet(batch_size=1, filename = 'van_gogh_starry_night.npy'):

    rng = np.random.RandomState(23455)

    input_img = np.load(filename).astype(np.float32)

    index = T.lscalar()

    x = T.ftensor3('x')

    print '... building the model'

    layer1_input = x.reshape((batch_size, 3, 227, 227))

    convpool_layer1 = ConvPoolLayer(input=layer1_input, image_shape=(batch_size, 3, 227, 227), filter_shape=(96, 3, 11, 11), f_params_w='W_0_65.npy', f_params_b='b_0_65.npy', lrn=True, convstride=4, padsize=0, group=1, poolsize=3, poolstride=2)

    convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,image_shape=(batch_size, 96, 27, 27),filter_shape=(256, 96, 5, 5), f_params_w=['W0_1_65.npy','W1_1_65.npy'], f_params_b=['b0_1_65.npy','b1_1_65.npy'], lrn=True, convstride=1, padsize=2, group=2, poolsize=3, poolstride=2)

    convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,image_shape=(batch_size, 256, 13, 13),filter_shape=(384, 256, 3, 3), f_params_w='W_2_65.npy', f_params_b='b_2_65.npy',convstride=1, padsize=1, group=1,poolsize=1, poolstride=0)

    convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,image_shape=(batch_size, 384, 13, 13),filter_shape=(384, 384, 3, 3), f_params_w=['W0_3_65.npy','W1_3_65.npy'], f_params_b=['b0_3_65.npy','b1_3_65.npy'],convstride=1, padsize=1, group=2,poolsize=1, poolstride=0)

    convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,image_shape=(batch_size, 384, 13, 13),filter_shape=(256, 384, 3, 3), f_params_w=['W0_4_65.npy','W1_4_65.npy'], f_params_b=['b0_4_65.npy','b1_4_65.npy'],convstride=1, padsize=1, group=2,poolsize=3, poolstride=2)

    inference_model = theano.function(
        [],
        [convpool_layer1.output, convpool_layer2.output, convpool_layer3.output, convpool_layer4.output, convpool_layer5.output],
        givens={
            x: input_img
        }
    )
    print('... inference')
    results = inference_model()

    for i in xrange(5):
        filestr = 'cnn_features/'+filename[:len(filename)-4]+'_%d.npy'%(i+1)
        np.save(filestr,results[i])
        print '%s saved'%filestr
        

if __name__ == '__main__':
    
    filenames = ['van_gogh_starry_night.npy', 'kaist_n1.npy']

    for f_idx in xrange(2):
        evaluate_alexnet(filename = filenames[f_idx])


def experiment(state, channel):
    evaluate_alexnet(state.learning_rate, dataset=state.dataset)
