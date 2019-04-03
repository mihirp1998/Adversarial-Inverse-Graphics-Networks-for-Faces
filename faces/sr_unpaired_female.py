# experiment: unpaired GAN with L2 loss on the low resolution images and
# only female images in the discriminator

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# this for sklearn 0.17, for 0.18: use sklearn.model_selection
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
import pdb

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])    
    restore_vars = []    
    for var_name, saved_var_name in var_names:            
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

# takes list of filenames and returns a 4D batch of images
# [N x W x H x C]
# also resize if necessary
def get_images(filenames, imsize=None):

    if imsize:
        batch_orig = [imresize(imread(path), (imsize, imsize), interp='bicubic') for path in filenames]
    else:
        batch_orig = [imread(path)for path in filenames]

    batch_orig_normed = np.array(batch_orig).astype(np.float32)/127.5-1

    batch_inputs = [imresize(im, 0.25, interp='bicubic') for im in batch_orig]
    # imresize returns in [0-255] so we have to normalize again
    batch_inputs_normed = np.array(batch_inputs).astype(np.float32)/127.5-1

    return batch_orig_normed, batch_inputs_normed

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def create_error_metrics(gen, inputs, origs):
    # Losses

    # metric: L2 between downsampled generated output and input
    gen_LR = slim.avg_pool2d(gen, [4, 4], stride=4, padding='SAME')
    gen_mse_LR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_LR - inputs)), 1)
    gen_L2_LR = tf.reduce_mean(gen_mse_LR)

    # metric: L2 between generated output and the original image
    gen_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen - origs)), 1)
    # average for the batch
    gen_L2_HR = tf.reduce_mean(gen_mse_HR)

    # metric: PSNR between generated output and original input
    gen_rmse_HR = tf.sqrt(gen_mse_HR)
    gen_PSNR = tf.reduce_mean(20*tf.log(1.0/gen_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))

    err_im_HR = gen - origs
    err_im_LR = gen_LR - inputs

    return gen_L2_LR, gen_L2_HR, gen_PSNR, err_im_LR, err_im_HR


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True, b_reuse=False):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                # huge hack
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                    self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                    with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)

        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed



########
# CONFIG 
#########
adam_learning_rate = 0.0001
adam_beta1 = 0.9

batch_size = 32
image_h = 128
image_w = 128

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################


real_ims = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='real_ims')

# the input to the generator is a downsampled version of the real image
inputs = tf.placeholder(tf.float32, [None, image_h/4, image_w/4, 3], name='inputs')


# generator section
print "GENERATOR"
print "-----------"

# this is really bad
batch_norm_list = []
nb_residual = 15
n_extra_bn = 1
for n in range(nb_residual*2 + n_extra_bn):
    batch_norm_list.append(batch_norm(name='bn'+str(n)))

def create_generator(inputs, b_training=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        net = tf.nn.relu(slim.conv2d(net, 64, [3, 3], scope='gconv1'))
        print_shape(net)

        net1 = net

        res_inputs = net1
        for n in range(nb_residual):
            net = tf.nn.relu(batch_norm_list[n*2](slim.conv2d(res_inputs, 64, [3, 3], scope='conv1_res'+str(n)), train=b_training))
            net = batch_norm_list[n*2+1](slim.conv2d(net, 64, [3, 3], scope='conv2_res'+str(n)), train=b_training)
            net = net + res_inputs
            res_inputs = net


        print_shape(net)

        net = batch_norm_list[-1](slim.conv2d(net, 64, [3, 3], scope='gconv2'), train=b_training) + net1
        print_shape(net)

        # deconv
        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv1'))
        print_shape(net)

        net = tf.nn.relu(slim.conv2d_transpose(net, 256, [5, 5], stride=2, scope='deconv2'))
        print_shape(net)


        # tanh since images have range [-1,1]
        net = slim.conv2d(net, 3, [3, 3], scope='gconv3', activation_fn=tf.nn.tanh)
        print_shape(net)

    return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs)
    scope.reuse_variables()
    gen_test = create_generator(inputs, False)


print "DISCRIMINATOR"
print "--------------"
#disc_bn1 = batch_norm(name='d_bn1')
disc_bn2 = batch_norm(name='d_bn2')
disc_bn3 = batch_norm(name='d_bn3')
disc_bn4 = batch_norm(name='d_bn4')
disc_bn5 = batch_norm(name='d_bn5')

# anneal our noise stddev
decay_counter = tf.Variable(0, name="counter", dtype=tf.float32)

def create_discriminator(inputs, counter, b_reuse=False):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    stride=2):

        # 0.05 starting stddev
        #noise_stddev = tf.constant(0.05, dtype=tf.float32) - tf.constant(5.5e-7, dtype=tf.float32)*counter
        #noisy_inputs = inputs + tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=noise_stddev, dtype=tf.float32)
        noisy_inputs = inputs

        disc = lrelu(slim.conv2d(noisy_inputs, 64, [5, 5], scope='conv1'))
        print_shape(disc)

        disc = lrelu(disc_bn2(slim.conv2d(disc, 128, [5, 5], scope='conv2'), b_reuse=b_reuse))
        print_shape(disc)

        disc = lrelu(disc_bn3(slim.conv2d(disc, 256, [5, 5], scope='conv3'), b_reuse=b_reuse))
        print_shape(disc)

        disc = lrelu(disc_bn4(slim.conv2d(disc, 512, [5, 5], scope='conv4'), b_reuse=b_reuse))
        print_shape(disc)

        disc = lrelu(disc_bn5(slim.conv2d(disc, 512, [5, 5], scope='conv5'), b_reuse=b_reuse))
        print_shape(disc)

    disc = lrelu(slim.fully_connected(tf.contrib.layers.flatten(disc), 1024, activation_fn=None, scope='fc6'))
    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc7')

    return disc_logits
# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims, decay_counter)
    scope.reuse_variables()
    disc_fake = create_discriminator(gen, decay_counter, True)


# Losses
###########

# loss on real input images; all outputs should be 1

# make labels noisy for discriminator
rand_val = tf.random_uniform([], seed=42)
labels_real = tf.cond(rand_val < 0.95, lambda: tf.ones_like(disc_real), lambda: tf.zeros_like(disc_real))
labels_fake = tf.cond(rand_val < 0.95, lambda: tf.zeros_like(disc_fake), lambda: tf.ones_like(disc_fake))

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))



gen_L2_LR, gen_L2_HR, gen_PSNR, err_im_LR, err_im_HR = create_error_metrics(gen, inputs, real_ims)

# metrics for testing stream
gen_L2_LR_t, gen_L2_HR_t, gen_PSNR_t, err_im_LR_t, err_im_HR_t = create_error_metrics(gen_test, inputs, real_ims)

# baselines: L2 and PSNR between bicubic upsampled input and original image
upsampled_output = tf.image.resize_bicubic(inputs, [image_h, image_h])
ups_mse_HR = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(upsampled_output - real_ims)), 1)
ups_L2_HR = tf.reduce_mean(ups_mse_HR)

ups_rmse_HR = tf.sqrt(ups_mse_HR)
ups_PSNR = tf.reduce_mean(20*tf.log(1.0/ups_rmse_HR)/tf.log(tf.constant(10, dtype=tf.float32)))


train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]


# optimize the generator and discriminator separately
g_loss = gen_L2_LR + 0.01*g_loss_adv
d_loss = d_loss_real + d_loss_fake

d_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(g_loss, var_list=g_vars)
    
weight_saver = tf.train.Saver(max_to_keep=1)


# logging

d_loss_train     = tf.summary.scalar("d_loss", d_loss)
g_L2LR_train     = tf.summary.scalar("g_loss_L2_LR", gen_L2_LR)
g_loss_adv_train = tf.summary.scalar("g_loss_adv", g_loss_adv)
g_L2HR_train     = tf.summary.scalar("gen_L2_HR", gen_L2_HR)
g_PSNR_train     = tf.summary.scalar("gen_PSNR_HR", gen_PSNR)

g_L2LR_test      = tf.summary.scalar("g_loss_L2_LR", gen_L2_LR_t)
g_L2HR_test      = tf.summary.scalar("gen_L2_HR", gen_L2_HR_t)
g_PSNR_test      = tf.summary.scalar("gen_PSNR_HR", gen_PSNR_t)

ups_L2HR = tf.summary.scalar("ups_L2_HR", ups_L2_HR)
ups_PSNR = tf.summary.scalar("ups_PSNR_HR", ups_PSNR)

merged_summary_train = tf.summary.merge([d_loss_train, g_L2LR_train, g_loss_adv_train, g_L2HR_train, g_PSNR_train, ups_L2HR, ups_PSNR])
merged_summary_test = tf.summary.merge([g_L2LR_test, g_L2HR_test, g_PSNR_test, ups_L2HR, ups_PSNR])
train_writer = tf.summary.FileWriter('./logs_female/train')
test_writer = tf.summary.FileWriter('./logs_female/test')



print "initialization done"


#############
# TRAINING
############

female_data_dir = '/home/wseto/datasets/celeba_female'
male_data_dir = '/home/wseto/datasets/celeba_male'

female_data = glob(os.path.join(female_data_dir, "*.png"))
male_data = glob(os.path.join(male_data_dir, "*.png"))

data_disc, female_data_nondisc = train_test_split(female_data, test_size=0.5, random_state=42)

female_data_train, female_data_sample = train_test_split(female_data_nondisc, test_size=0.1, random_state=42)
male_data_train, male_data_sample = train_test_split(male_data, test_size=0.1, random_state=42)

data_train = female_data_train + male_data_train
data_sample = female_data_sample + male_data_sample


print "data train:", len(data_train)
print "data disc:", len(data_disc)
print "data sample:", len(data_sample)

# create directories to save checkpoint and samples
samples_dir = 'samples_female'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_female'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0

b_load = False
#ckpt_dir = '/home/wseto/dcgan/checkpoint_up_rand'

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    if b_load:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        weight_saver.restore(sess, ckpt.model_checkpoint_path)
        counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
        print "successfully restored!" + " counter:", counter
        
    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        total_errD = 2

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            
            batch_origs, batch_inputs = get_images(batch_filenames)
            
            # discriminator batch is different since we are doing unpaired experiment
            rand_idx = np.random.randint(len(data_disc)-batch_size-1)
            disc_batch_files = data_disc[rand_idx: rand_idx+batch_size]     
            disc_batch_orig, disc_batch_inputs = get_images(disc_batch_files)


            # errD_fake = d_loss_fake.eval({inputs: batch_inputs})
            # errD_real = d_loss_real.eval({real_ims: disc_batch_orig})
            # errG = g_loss.eval({ inputs: batch_inputs, real_ims: disc_batch_orig})

            if total_errD > 1:
                fetches = [d_loss_fake, d_loss_real, g_loss_adv, d_optim, g_optim]
                errD_fake, errD_real, errG, _, _ = sess.run(fetches, feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})
            else:
                fetches = [d_loss_fake, d_loss_real, g_loss_adv, g_optim]
                errD_fake, errD_real, errG, _ = sess.run(fetches, feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig})

            total_errD = errD_fake + errD_real



            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_fake: %.3f, d_loss_real: %.3f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake, errD_real, errG))


            if np.mod(counter, 30) == 1:

                 # training metrics first
                train_summary = sess.run([merged_summary_train], feed_dict={ inputs: batch_inputs, real_ims: batch_origs})
                train_writer.add_summary(train_summary[0], counter)

                # now testing metrics
                rand_idx = np.random.randint(len(data_sample)-batch_size+1)
                sample_origs, sample_inputs = get_images(data_sample[rand_idx: rand_idx+batch_size])

                sample = sess.run([gen_test], feed_dict={inputs: sample_inputs})

                err_im_LR = sess.run([err_im_LR_t], feed_dict={inputs: sample_inputs})
                resz_err_im = err_im_LR[0][0].repeat(axis=0,repeats=4).repeat(axis=1,repeats=4)

                test_summary = sess.run([merged_summary_test], feed_dict={ inputs: sample_inputs, real_ims: sample_origs})
                test_writer.add_summary(test_summary[0], counter)

                # save an image, with the original next to the generated one
                resz_input = sample_inputs[0].repeat(axis=0,repeats=4).repeat(axis=1,repeats=4)
                merge_im = np.zeros( (image_h, image_h*4, 3) )
                merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
                merge_im[:, image_h:image_h*2, :] = (resz_input+1)*127.5
                merge_im[:, image_h*2:image_h*3, :] = (sample[0][0]+1)*127.5
                merge_im[:, image_h*3:, :] = (resz_err_im+1)*127.5

                imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, checkpoint_dir + '/model', counter)
                print "saving a checkpoint"