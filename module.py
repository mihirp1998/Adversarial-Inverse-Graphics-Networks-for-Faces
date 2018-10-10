from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator(image,options,is_training, reuse=False, name="discriminator"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False    

        h0 = lrelu(conv2dn(image, 64, name='d_h0_conv'))

        h1 = lrelu(batch_normn(name='d_bn1')(conv2dn(h0, 64*2, name='d_h1_conv'),is_training))
        h2 = lrelu(batch_normn(name='d_bn2')(conv2dn(h1, 64*4, name='d_h2_conv'),is_training))
        h3 = lrelu(batch_normn(name='d_bn3')(conv2dn(h2, 64*8, name='d_h3_conv'),is_training))
        h3 = lrelu(batch_normn(name='d_bn4')(conv2dn(h3, 64*8, name='d_h4_conv'),is_training))

        print(h3)
        flat = tf.layers.flatten(h3,name="flatten")
        print(flat)
        h4 = linear(flat, 1, 'd_h4_lin')
        h4 = tf.nn.dropout(h4,0.5)


        return h4



def generator_resnet(image, options,is_training, reuse=False, name="generator"):

    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            y = conv2dn(x,dim, k_h=5, k_w=5, d_h=1, d_w=1, name=name+'_c1')
            y = batch_normn(name = name+'_bn1')(y,is_training)
            y = tf.nn.relu(y)
            y = conv2dn(y,dim, k_h=5, k_w=5, d_h=1, d_w=1, name=name+'_c2')
            y = batch_normn(name = name+'_bn2')(y,is_training)

            return y + x


        c1 = tf.nn.relu(batch_normn(name = 'init_bn')(conv2dn(image,64, k_h=5, k_w=5, d_h=1, d_w=1, name='init_conv'),is_training))

        
        r1 = residule_block(c1, 64, name='g_r1')
        r2 = residule_block(r1, 64, name='g_r2')
        r3 = residule_block(r2, 64, name='g_r3')
        r4 = residule_block(r3, 64, name='g_r4')
        r5 = residule_block(r4, 64, name='g_r5')
        r6 = residule_block(r5, 64, name='g_r6')
        r7 = residule_block(r6, 64, name='g_r7')
        r8 = residule_block(r7, 64, name='g_r8')
        r9 = residule_block(r8, 64, name='g_r9')

        c2 = tf.nn.relu(batch_normn(name ='first_bn')(conv2dn(image, 64, k_h=5, k_w=5, d_h=1, d_w=1, name='first_conv'),is_training))
        d1 = deconv2dn(c2,[tf.shape(c2)[0],64,64,256], name='g_d1_dc1')
        d1 = batch_normn(name = name+'_bn3')(d1,is_training)
        d1 = tf.nn.relu(d1)
        d2 = deconv2dn(d1,[tf.shape(d1)[0],128,128,256], name='g_d2_dc')
        d2 = batch_normn(name = name+'_bn4')(d2,is_training)
        d2 = tf.nn.relu(d2)
        pred = tf.nn.tanh(conv2dn(d2, 3, k_h=5, k_w=5, d_h=1, d_w=1,name='g_pred_c'))

        return pred



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
