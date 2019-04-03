

from scipy.misc import imread, imshow, imresize, imsave
import os
from glob import glob
import numpy as np
# for below 0.18, use sklearn.cross_validation
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle


with open("nose_bb64.pickle", "rb") as fp:
    mouth_bb_dict = pickle.load(fp)


def print_shape(t):
    print(t.name, t.get_shape().as_list())

def get_images(filenames, imsize=None):

    if imsize:
        batch_orig = [imresize(imread(path), (imsize, imsize), interp='bicubic') for path in filenames]
    else:
        batch_orig = [imread(path)for path in filenames]

    batch_orig_normed = np.array(batch_orig).astype(np.float32)/127.5-1

    # create masked inputs
    batch_inputs = []
    masks = []
    for idx in range(len(filenames)):

        mask = np.ones((imsize, imsize, 1))
        image_filename = filenames[idx].split("/")[-1]
        mouth_bb = mouth_bb_dict[image_filename]

        for x in range(mouth_bb[2]):
            for y in range(mouth_bb[3]):
                mask[mouth_bb[1]+y, mouth_bb[0]+x, 0] = 0

        #print mask.shape
        batch_inputs.append(batch_orig_normed[idx] * mask)
        masks.append(mask)

    return batch_orig_normed, batch_inputs, masks

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name, is_training=train)




########
# CONFIG 
#########
adam_learning_rate = 0.0002
adam_beta1 = 0.9

batch_size = 64
image_h = 64
image_w = 64

num_epochs = 20

###############################
# BUILDING THE MODEL
###############################


real_ims = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='real_ims')

inputs = tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='inputs')

input_masks = tf.placeholder(tf.float32, [None, image_h, image_w, 1], name='input_masks')


# generator section
print "GENERATOR"
print "-----------"

gen_bn1 = batch_norm(name='g_bn1')
gen_bn2 = batch_norm(name='g_bn2')
gen_bn3 = batch_norm(name='g_bn3')

gen_bn_m1 = batch_norm(name='g_bn_m1')
gen_bn_m2 = batch_norm(name='g_bn_m2')

def create_generator(inputs, input_masks, b_train=True):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        padding='SAME',
                        activation_fn=None,
                        stride=2,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = inputs
        print_shape(net)

        # encoder section
        net = tf.nn.relu(gen_bn1(slim.conv2d(net, 256, [3, 3], scope='gconv1'), train=b_train))
        print_shape(net)

        net = tf.nn.relu(gen_bn2(slim.conv2d(net, 1024, [3, 3], scope='gconv2'), train=b_train))
        print_shape(net)


        # process input masks
        net_mask = tf.nn.relu(gen_bn_m1(slim.conv2d(input_masks, 256, [3, 3], scope='gconv_m1'), train=b_train))
        net_mask = tf.nn.relu(gen_bn_m2(slim.conv2d(net_mask, 1024, [3, 3], scope='gconv_m2'), train=b_train))


        net = tf.concat([net_mask, net], 3)
        #print_shape(net)

        # decoder section
        net = tf.nn.relu(gen_bn3(slim.conv2d_transpose(net, 512, [5, 5], scope='deconv3'), train=b_train))
        print_shape(net)

        # tanh since images have range [-1,1]
        net = slim.conv2d_transpose(net, 3, [5, 5], scope='deconv4', activation_fn=tf.nn.tanh)
        print_shape(net)

        return net

with tf.variable_scope("generator") as scope:
    gen = create_generator(inputs, input_masks)
    scope.reuse_variables()
    gen_test = create_generator(inputs, input_masks, False)

    
print "DISCRIMINATOR"
print "--------------"

disc_bn1 = batch_norm(name='d_bn1')
disc_bn2 = batch_norm(name='d_bn2')
disc_bn3 = batch_norm(name='d_bn3')

def create_discriminator(inputs):
    with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    activation_fn=None,
                    stride=2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
        disc = lrelu(slim.conv2d(inputs, 64, [5, 5], scope='dconv1'))
        print_shape(disc)

        disc = lrelu(disc_bn1(slim.conv2d(disc, 128, [5, 5], scope='dconv2')))
        print_shape(disc)

        disc = lrelu(disc_bn2(slim.conv2d(disc, 256, [5, 5], scope='dconv3')))
        print_shape(disc)

        disc = lrelu(disc_bn3(slim.conv2d(disc, 512, [5, 5], scope='dconv4')))
        print_shape(disc)

    disc_logits = slim.fully_connected(tf.contrib.layers.flatten(disc), 1, activation_fn=None, scope='fc5')
    return disc_logits

# create 2 discriminators: for fake and real images
with tf.variable_scope("discriminators") as scope:
    disc_real = create_discriminator(real_ims)
    scope.reuse_variables()
    disc_fake = create_discriminator(gen)
    

# Losses
###########

# loss on real input images; all outputs should be 1

# make labels noisy for discriminator
rand_val = tf.random_uniform([], seed=42)
labels_real = tf.cond(rand_val < 1, lambda: tf.ones_like(disc_real), lambda: tf.zeros_like(disc_real))
labels_fake = tf.cond(rand_val < 1, lambda: tf.zeros_like(disc_fake), lambda: tf.ones_like(disc_fake))

d_loss_real = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=labels_real))

# loss on fake input images, fakes should be 0
d_loss_fake = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=labels_fake))

# similar to above, but we want fake (generator) images to output 1
g_loss_adv = tf.reduce_mean( \
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))



# Losses

# metric: L2 between non masked parts of image
gen_masked = tf.multiply(gen, input_masks)
input_masked = tf.multiply(inputs, input_masks)
gen_mse = tf.reduce_mean(tf.square(tf.contrib.layers.flatten(gen_masked - input_masked)), 1)
# average for the batch
gen_L2 = tf.reduce_mean(gen_mse)

# metric: PSNR between generated output and original input
gen_rmse = tf.sqrt(gen_mse)
gen_PSNR = tf.reduce_mean(20*tf.log(1.0/gen_rmse)/tf.log(tf.constant(10, dtype=tf.float32)))



train_vars = tf.trainable_variables()

d_vars = [var for var in train_vars if 'discriminators' in var.name]
g_vars = [var for var in train_vars if 'generator' in var.name]

g_loss = gen_L2 + 0.001*g_loss_adv
d_loss = d_loss_real + d_loss_fake


# optimizer the generator and discriminator separately
d_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(adam_learning_rate, beta1=adam_beta1) \
                  .minimize(g_loss, var_list=g_vars)
    
weight_saver = tf.train.Saver(max_to_keep=1)

# logging

tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("g_loss_adv", g_loss_adv)

tf.summary.scalar("gen_L2_HR", gen_L2)
tf.summary.scalar("gen_PSNR", gen_PSNR)


merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs_ip_nose/train')
test_writer = tf.summary.FileWriter('./logs_ip_nose/test')

print "initialization done"


#############
# TRAINING
############


bnose_data_dir = '/home/wseto/datasets/celeba_bnose'
snose_data_dir = '/home/wseto/datasets/celeba_snose'

bnose_data = glob(os.path.join(bnose_data_dir, "*.png"))
snose_data = glob(os.path.join(snose_data_dir, "*.png"))

data_disc, bnose_data_nondisc = train_test_split(bnose_data, test_size=0.1, random_state=42)

bnose_data_train, bnose_data_sample = train_test_split(bnose_data_nondisc, test_size=0.5, random_state=42)
snose_data_train, snose_data_sample = train_test_split(snose_data, test_size=0.08, random_state=42)

data_train = bnose_data_train + snose_data_train
data_sample = bnose_data_sample + snose_data_sample

print "data train:", len(data_train)
print "data disc:", len(data_disc)
print "data sample:", len(data_sample)

# create directories to save checkpoint and samples
samples_dir = 'samples_ip_nose'
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

checkpoint_dir = 'checkpoint_ip_nose'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print "TRAINING"
print "-----------"

start_time = time.time()
counter = 0
b_load = False

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    num_batches = len(data_train) // batch_size

    if b_load:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        weight_saver.restore(sess, ckpt.model_checkpoint_path)
        counter = int(ckpt.model_checkpoint_path.split('-', 1)[1]) 
        print "successfuly restored!" + " counter:", counter

    for epoch in range(num_epochs):

        np.random.shuffle(data_train)

        for idx in xrange(num_batches):
            batch_filenames = data_train[idx*batch_size : (idx+1)*batch_size]
            
            batch_origs, batch_inputs, batch_masks = get_images(batch_filenames, imsize=image_h)
            
            # discriminator batch is different since we are doing unpaired experiment
            rand_idx = np.random.randint(len(data_disc)-batch_size-1)
            disc_batch_files = data_disc[rand_idx: rand_idx+batch_size]     
            disc_batch_orig, disc_batch_inputs, _ = get_images(disc_batch_files, imsize=image_h)

            fetches = [d_loss_fake, d_loss_real, g_loss_adv, d_optim, g_optim]
            errD_fake, errD_real, errG, _, _ = sess.run(fetches, feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig, input_masks: batch_masks})

            # Run g_optim twice to make sure that d_loss does not go to zero
            sess.run([g_optim], feed_dict={ inputs: batch_inputs, real_ims: disc_batch_orig, input_masks: batch_masks})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, idx, num_batches,
                    time.time() - start_time, errD_fake+errD_real, errG))



            if np.mod(counter, 30) == 1:

                rand_idx = np.random.randint(len(data_sample)-batch_size+1)
                sample_origs, sample_inputs, sample_masks = get_images(data_sample[rand_idx: rand_idx+batch_size], imsize=image_h)

                sample, loss = sess.run([gen_test, g_loss], feed_dict={inputs: sample_inputs, real_ims: disc_batch_orig, input_masks: sample_masks})
                print "Sample loss: ", loss

                test_summary = sess.run([merged_summary], feed_dict={ inputs: sample_inputs, real_ims: sample_origs, input_masks: sample_masks})
                test_writer.add_summary(test_summary[0], counter)


                sample = [sample]
                # save an image, with the original next to the generated one
                merge_im = np.zeros( (image_h, image_h*4, 3) )
                merge_im[:, :image_h, :] = (sample_origs[0]+1)*127.5
                merge_im[:, image_h:image_h*2, :] = (sample_inputs[0]+1)*127.5
                merge_im[:, image_h*2:image_h*3, :] = (sample[0][0]+1)*127.5

                # final, final output: take the masked region from the generator output 
                # and put it together with our original masked input

                final_output = sample_inputs[0] + (1-sample_masks[0])*sample[0][0]
                merge_im[:, image_h*3:, :] = (final_output+1)*127.5
                imsave(samples_dir + '/test_{:02d}_{:04d}.png'.format(epoch, idx), merge_im)

            if np.mod(counter, 1000) == 2:
                weight_saver.save(sess, checkpoint_dir + '/model', counter)
                print "saving a checkpoint"

