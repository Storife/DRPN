# coding: utf-8
__author__ = 'zq'


import lasagne
import theano
import theano.tensor as T
import numpy as np
import cPickle
import gzip
from test import Data_getPicNameList, Fun_test
from lasagne.layers import InputLayer, Upscale2DLayer, InverseLayer, ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer, Upscale2DLayer
from lasagne.layers import NonlinearityLayer, BatchNormLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.objectives import binary_crossentropy
# from collections import OrderedDict

BATCH_SIZE = 8
Input_Shape = [[None, 3, 256, 256]]
Output_Shape = [[None, 1, 256, 256]]
saveParamName= 'DRPN.pkl.gz'
# LEARNING_RATE = 0.001
# MOMENTUM = 0.9




def main():

    print("Building model and compiling functions...")
    X_batch = [T.tensor4('x')]
    y_batch = [T.tensor4('y')]

    net = {}
    net['input'] = InputLayer((None, 3, 256, 256), input_var= X_batch[0])
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)

    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)

    initSal = {}
    initSal['up'] = Upscale2DLayer(net['conv5_3'], (2, 2))
    initSal['concat'] = ConcatLayer([initSal['up'], net['conv4_3']])

    initSal['conv1'] = ConvLayer(initSal['concat'], 1024, (3, 3), pad=1, flip_filters=False)
    initSal['conv2'] = ConvLayer(initSal['conv1'], 512, (1, 1), pad=0, flip_filters=False)
    initSal['conv3'] = ConvLayer(initSal['conv2'], 256, (5, 5), pad=2, flip_filters=False)
    initSal['output'] = ConvLayer(initSal['conv3'], 1, (1, 1), nonlinearity=lasagne.nonlinearities.sigmoid, flip_filters=False)

    # ***************************************************************************************************
    recurent = {}
    recurent['1-sal'] = BatchNormLayer(Upscale2DLayer(initSal['conv3'], (2, 2)))
    recurent['1-vgg'] = BatchNormLayer(ConvLayer(net['conv3_3'], 256, 1))
    recurent['1-input'] = ConcatLayer([recurent['1-sal'], recurent['1-vgg']])

    recurent['1-NIN1'] = ConvLayer(recurent['1-input'], 256, 1, pad=0, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['1-NIN1'] = NonlinearityLayer(BatchNormLayer(recurent['1-NIN1']))
    recurent['1-rconv1'] = ConvLayer(recurent['1-NIN1'], 256, 3, pad=1, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['1-rconv1'] = NonlinearityLayer(BatchNormLayer(recurent['1-rconv1']))
    recurent['1-rconv2'] = ConvLayer(recurent['1-rconv1'], 256, 3, pad=1, flip_filters=False)
    recurent['1-rconv2'] = NonlinearityLayer(BatchNormLayer(recurent['1-rconv2']))
    recurent['1-rconv3'] = ConvLayer(recurent['1-rconv2'], 256, 1, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['1-sum']    = ElemwiseSumLayer([recurent['1-rconv3'], recurent['1-sal']])
    recurent['1-sum']    = NonlinearityLayer(recurent['1-sum'])
    # ***************************************************************************************************
    recurent['2-sal'] = BatchNormLayer(Upscale2DLayer(recurent['1-sum'], (2, 2)))
    recurent['2-vgg'] = BatchNormLayer(ConvLayer(net['conv2_2'], 256, 1))
    recurent['2-input'] = ConcatLayer([recurent['2-sal'], recurent['2-vgg']])

    recurent['2-NIN1'] = ConvLayer(recurent['2-input'], 256, 1, pad=0, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['2-NIN1'] = NonlinearityLayer(BatchNormLayer(recurent['2-NIN1']))
    recurent['2-rconv1'] = ConvLayer(recurent['2-NIN1'], 256, 3, pad=1, flip_filters=False)
    recurent['2-rconv1'] = NonlinearityLayer(BatchNormLayer(recurent['2-rconv1']))
    recurent['2-rconv2'] = ConvLayer(recurent['2-rconv1'], 256, 3, pad=1, flip_filters=False)
    recurent['2-rconv2'] = NonlinearityLayer(BatchNormLayer(recurent['2-rconv2']))
    recurent['2-rconv3'] = ConvLayer(recurent['2-rconv2'], 256, 1, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['2-sum']    = ElemwiseSumLayer([recurent['2-rconv3'], recurent['2-sal']])
    recurent['2-sum']    = NonlinearityLayer(recurent['2-sum'])
    # ***************************************************************************************************
    recurent['3-sal'] = BatchNormLayer(Upscale2DLayer(recurent['2-sum'], (2, 2)))
    recurent['3-vgg'] = ConvLayer(net['conv1_2'], 128, 3, pad=1)
    recurent['3-vgg'] = BatchNormLayer(ConvLayer(recurent['3-vgg'], 256, 1))
    recurent['3-input'] = ConcatLayer([recurent['3-sal'], recurent['3-vgg']])
    recurent['3-NIN1'] = ConvLayer(recurent['3-input'], 256, 1, pad=0, flip_filters=False)
    recurent['3-NIN1'] = NonlinearityLayer(BatchNormLayer(recurent['3-NIN1']))
    recurent['3-rconv1'] = ConvLayer(recurent['3-NIN1'], 256, 3, pad=1, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['3-rconv1'] = NonlinearityLayer(BatchNormLayer(recurent['3-rconv1']))
    recurent['3-rconv2'] = ConvLayer(recurent['3-rconv1'], 256, 3, pad=1, flip_filters=False)
    recurent['3-rconv2'] = NonlinearityLayer(BatchNormLayer(recurent['3-rconv2']))
    recurent['3-rconv3'] = ConvLayer(recurent['3-rconv2'], 256, 1, nonlinearity=lasagne.nonlinearities.linear, flip_filters=False)
    recurent['3-sum']    = ElemwiseSumLayer([recurent['3-rconv3'], recurent['3-sal']])
    recurent['3-sum']    = NonlinearityLayer(recurent['3-sum'])
    recurent['3-output'] = ConvLayer(recurent['3-sum'], 1, (1, 1), nonlinearity=lasagne.nonlinearities.sigmoid, flip_filters=False)



    prediction = []
    # loss_train = []
    # all_params = []
    # accuracy = []
    output_layer = recurent['3-output']
    prediction.append(lasagne.layers.get_output(output_layer))
    # loss_train.append(T.mean(lasagne.objectives.binary_crossentropy(prediction[0], y_batch[0])))
    # all_params.append(lasagne.layers.get_all_params(output_layer, trainable=True))
    # accuracy.append(T.mean(T.square(prediction[0]-y_batch[0]))),
    # updates = OrderedDict()
    # update = lasagne.updates.sgd(loss_train[0], all_params[0][32::], LEARNING_RATE)
    # updates.update(update)
    # updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=MOMENTUM)

    FitSetting = dict(
        Input_Shape=Input_Shape,
        Output_Shape=Output_Shape,
        output_layer=output_layer,
        saveParamName=saveParamName,
    )
    output_layer = [output_layer]
    output = lasagne.layers.get_output(output_layer)
    Fun_test(['./image/'], ['.jpg'], FitSetting, X_batch, output, writeimg=['DRPN'])
if __name__ == '__main__':
    main()
    # main()
