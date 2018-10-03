import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

from utils import *

def batch_norm(x,is_training, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name,is_training= is_training)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

# def conv2d(input_, output_dim, ks=4, s=2, stddev=0.141, padding='SAME', name="conv2d",use_bias= True):
#     with tf.variable_scope(name):
#         return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
#                             weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
#                             biases_initializer=None)


def conv2d(input_, output_dim, ks=7, s=2, stddev=0.141, padding='SAME', name="conv2d",use_bias= True):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev))

def conv2dn(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    # print("before bias",conv)
    conv = tf.nn.bias_add(conv, biases)
    # print("after bias",conv)
    return conv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  print("shape is ",shape)
  with tf.variable_scope(scope or "Linear"):
    try:
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
    except ValueError as err:
        msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
        err.args = err.args + (msg,)
        raise
    bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.141, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)


def deconv2dn(input_, output_shape,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.nn.bias_add(deconv, biases)

    if with_w:
        return deconv, w, biases
    else:
        return deconv        

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


class batch_normn(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,scope=self.name)    

# def linear(input_, output_size, scope=None, stddev=0.141, bias_start=0.0, with_w=False):

#     with tf.variable_scope(scope or "Linear"):
#         matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=stddev))
#         bias = tf.get_variable("bias", [output_size],
#             initializer=tf.constant_initializer(bias_start))
#         if with_w:
#             return tf.matmul(input_, matrix) + bias, matrix, bias
#         else:
#             return tf.matmul(input_, matrix) + bias
