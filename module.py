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
         # image is (128 x 128 x 3)

        h0 = lrelu(conv2dn(image, 64, name='d_h0_conv'))
         # h0 is (64 x 64 x 64)

        h1 = lrelu(batch_normn(name='d_bn1')(conv2dn(h0, 64*2, name='d_h1_conv'),is_training))
         # h1 is (32 x 32 x 128)

        h2 = lrelu(batch_normn(name='d_bn2')(conv2dn(h1, 64*4, name='d_h2_conv'),is_training))
         # h2 is (16 x 16 x 256)

        h3 = lrelu(batch_normn(name='d_bn3')(conv2dn(h2, 64*8, name='d_h3_conv'),is_training))
         # h3 is (8 x 8 x 512)

        h4 = lrelu(batch_normn(name='d_bn4')(conv2dn(h3, 64*8, name='d_h4_conv'),is_training))
         # h4 is (4 x 4 x 512)

        flat = tf.layers.flatten(h4,name="flatten")
         # flat is (8192)

        dflat = tf.nn.dropout(flat,0.5)

         # dropout (4096)
          
        pred = linear(dflat, 1, 'd_h4_lin')
         # pred (1) 

        return pred



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

         # image is (32 x 32 x 3)
        c1 = tf.nn.relu(batch_normn(name = 'init_bn')(conv2dn(image,64, k_h=5, k_w=5, d_h=1, d_w=1, name='init_conv'),is_training))
         # c1 is (32 x 32 x 64)

        
        r1 = residule_block(c1, 64, name='g_r1')
        r2 = residule_block(r1, 64, name='g_r2')
        r3 = residule_block(r2, 64, name='g_r3')
        r4 = residule_block(r3, 64, name='g_r4')
        r5 = residule_block(r4, 64, name='g_r5')
        r6 = residule_block(r5, 64, name='g_r6')
        r7 = residule_block(r6, 64, name='g_r7')
        r8 = residule_block(r7, 64, name='g_r8')
        r9 = residule_block(r8, 64, name='g_r9')
         # r1 to r9 is (32 x 32 x 64)


        c2 = tf.nn.relu(batch_normn(name ='first_bn')(conv2dn(image, 64, k_h=5, k_w=5, d_h=1, d_w=1, name='first_conv'),is_training))
         # c2 is (32 x 32 x 64)

        d1 = deconv2dn(c2,[tf.shape(c2)[0],64,64,256], name='g_d1_dc1')
         # d1 is (64 x 64 x 256)

        d1 = batch_normn(name = name+'_bn3')(d1,is_training)
        d1 = tf.nn.relu(d1)
        d2 = deconv2dn(d1,[tf.shape(d1)[0],128,128,256], name='g_d2_dc')
         # d2 is (128 x 128 x 256)

        d2 = batch_normn(name = name+'_bn4')(d2,is_training)
        d2 = tf.nn.relu(d2)
        pred = tf.nn.tanh(conv2dn(d2, 3, k_h=5, k_w=5, d_h=1, d_w=1,name='g_pred_c'))
         # pred is (128 x 128 x 3)

        return pred



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
