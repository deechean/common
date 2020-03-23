#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:37:08 2019

@author: Deechean
"""

import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import numpy as np
import time
#import aws_boto3

def get_variable(name, shape, initializer, regularizer=None, dtype='float', trainable=True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           collections=collections,
                           dtype=dtype,
                           trainable=trainable)
    #tf.get_variable_scope().reuse_variables()

def conv2d(x, ksize, stride, filter_out, name, padding='VALID', activate = 'NONE'):
    """ 
    x: input 
    ksize: kernel size 
    stride
    filter_out: filters numbers
    name: name of the calculation
    padding: VALID - no padding, SAME - keep the output size same as input size
    activate: RELU - relu or SIGMOID  -sigmoid, TANH - tanh
    """
    with tf.variable_scope(name):
        #Get input dimention
        filter_in = x.get_shape()[-1]        
        stddev = 1. / tf.sqrt(tf.cast(filter_out, tf.float32))
        
        #use random uniform to initialize weight
        weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        
        #use random uniform to initialize bias
        bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        
        #kernel shape is [kenel size, kernel size, filter in size, filter out size]
        shape = [ksize, ksize, filter_in, filter_out]
        
        #set kernel
        kernel = get_variable('kernel', shape, weight_initializer)
        
        #set bias, bias shape is [filter_out]
        bias = get_variable('bias', [filter_out], bias_initializer)
        
        #conv2d
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding, name='conv')
        
        #add conv result with bias
        out = tf.nn.bias_add(conv, bias)
        
        #activate           
        if activate == 'SIGMOID':
            out = tf.nn.sigmoid(out)
        elif activate == 'TANH':
            out = tf.nn.tanh(out)
        elif activate == 'RELU':
            out = tf.nn.relu(out)
        return out
    
    
def max_pool(x, ksize, stride, name, padding):
    """ x: input
        ksize: kernel size
        stride: stride
        name: name of the calculation
        padding: VALID - no padding, SAME - keep the output size same as input size
    """
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], name=name, padding=padding)

def avg_pool(x, ksize, stride, name, padding):
    """ average pool
        x: input
        ksize: kernel size
        stride: stride
        name: name of the calculation
        padding: VALID - no padding, SAME - keep the output size same as input size
    """    
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1],[1, stride, stride, 1], name=name, padding=padding)

def flatten(x):
    """Reshape x to a list(one dimesion)
    """    
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1, len(shape)):
        dim *= shape[i]
    return tf.reshape(x, [-1, dim]), dim

def fc_layer(x, i_size, o_size, name, activate = 'NONE'):
    """Full connection layer
        x:
        i_size: input size
        o_size: output size
        name: name of the calculation
        activate: RELU - relu or SIGMOID  -sigmoid, TANH - tanh
    """
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[i_size, o_size], dtype='float')
        b = tf.get_variable('b', shape=[o_size], dtype='float')
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        
         #activate           
        if activate == 'SIGMOID':
            out = tf.nn.sigmoid(out)
        elif activate == 'TANH':
            out = tf.nn.tanh(out)
        elif activate == 'RELU':
            out = tf.nn.relu(out)
    
        return out

def drop_out(x, drop_rate, name):
    """drop out to prevent overfit, it should only used in training, not in test
        x: input
        drop_rate: probability of drop out, normally is 0.5
        name: name of the calculation
        
    """
    if tf.__version__ >= '1.13.0':
        return tf.nn.dropout(x, rate=drop_rate, name=name)
    else:
        return tf.nn.dropout(x, keep_prob=1-drop_rate, name=name)
    
    
    
def printimages(images):
    for img in images:    
        plt.imshow(np.asarray(img).reshape(32,32,3))
        plt.show()

def SaveCheckpoint2S3(ori_dir,des_dir):
    bucketname = 'sagemaker-deechean-dl' 
    for file in os.listdir(file_dir):
        if os.path.isfile(file_dir+file):
            print('save files to s3')
            aws_boto3.upload_file(file_dir+file, bucketname,des_dir+file)