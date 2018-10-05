from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator(image,options,is_training, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False    
        # h0 = lrelu(batch_norm(conv2d(c1,64, name='d_h0_conv',use_bias=False),is_training, 'd_bn0'))
        # # h0 is (128 x 128 x self.df_dim)
        # h1 = lrelu(batch_norm(conv2d(h0,64*2, name='d_h1_conv',use_bias=False),is_training, 'd_bn1'))
        # # h1 is (64 x 64 x self.df_dim*2)
        # h2 = lrelu(batch_norm(conv2d(h1, 64*4, name='d_h2_conv',use_bias=False),is_training, 'd_bn2'))
        # # h2 is (32x 32 x self.df_dim*4)                                        
        # h3 = lrelu(batch_norm(conv2d(h2, 64*8, name='d_h3_conv',use_bias=False),is_training, 'd_bn3'))
        # # h3 is (32 x 32 x self.df_dim*8)
        # h3 = lrelu(batch_norm(conv2d(h3, 64*8, name='d_h4_conv',use_bias=False),is_training, 'd_bn4'))
        # # print(h3)
        # h3_flat = tf.layers.flatten(h3,name="flatten")

        h0 = lrelu(conv2dn(image, 64, name='d_h0_conv'))

        h1 = lrelu(batch_normn(name='d_bn1')(conv2dn(h0, 64*2, name='d_h1_conv'),is_training))
        h2 = lrelu(batch_normn(name='d_bn2')(conv2dn(h1, 64*4, name='d_h2_conv'),is_training))
        h3 = lrelu(batch_normn(name='d_bn3')(conv2dn(h2, 64*8, name='d_h3_conv'),is_training))
        h3 = lrelu(batch_normn(name='d_bn4')(conv2dn(h3, 64*8, name='d_h4_conv'),is_training))

        print(h3)
        flat = tf.layers.flatten(h3,name="flatten")
        print(flat)
        h4 = linear(flat, 1, 'd_h4_lin')
        tf.nn.dropout(h4,0.5)

        # logit = tf.layers.dense(h3_flat,1,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1414))
        # logit = batch_norm(logit,is_training, 'd_bn5')
        # logit = tf.nn.relu(logit)
        return h4

def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


# def generator_resnet(image, options,is_training, reuse=False, name="generator"):

#     with tf.variable_scope(name):
#         # image is 256 x 256 x input_c_dim
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse is False

#         def residule_block(x, dim, ks=3, s=1, name='res'):
#             y = conv2d(x,dim, k_h=5, k_w=5, d_h=1, d_w=1, name=name+'_c1')
#             y = batch_norm(y,is_training, name+'_bn1')
#             y = tf.nn.relu(y)
#             y = conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2')
#             y = batch_norm(y,is_training,name+ '_bn2')
#             return y + x


#         c1 = tf.nn.relu(batch_norm(conv2d(image, 64, 7, 1, padding='SAME', name='init_conv',use_bias=False),is_training, 'init_bn'))

        
#         r1 = residule_block(c1, 64, name='g_r1')
#         r2 = residule_block(r1, 64, name='g_r2')
#         r3 = residule_block(r2, 64, name='g_r3')
#         r4 = residule_block(r3, 64, name='g_r4')
#         r5 = residule_block(r4, 64, name='g_r5')
#         r6 = residule_block(r5, 64, name='g_r6')
#         r7 = residule_block(r6, 64, name='g_r7')
#         r8 = residule_block(r7, 64, name='g_r8')
#         r9 = residule_block(r8, 64, name='g_r9')

#         c2 = tf.nn.relu(batch_norm(conv2d(image, 64, 3, 1, padding='SAME', name='first_conv',use_bias=False),is_training,'first_bn'))
#         d1 = deconv2d(c2, 256, 3, 2, name='g_d1_dc1')
#         d1 = tf.nn.relu(d1)
#         d2 = deconv2d(d1, 256, 3, 2, name='g_d2_dc')
#         d2 = tf.nn.relu(d2)

#         pred = tf.nn.tanh(conv2d(d2, 3, 4, 1, padding='SAME', name='g_pred_c'))

#         return pred


def generator_resnet(image, options,is_training, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
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
